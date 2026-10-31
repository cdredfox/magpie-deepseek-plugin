// 单元测试：bun test.mjs
// 覆盖：messages 拍平（含裁剪/增量拆分）、DSML 工具协议（解析+流式筛分）、
// SSE 转写（含 hint/ready 事件）、会话复用、错误分级重试、鉴权映射。
import assert from "node:assert/strict";
import {
  buildPrompt,
  buildPromptParts,
  buildToolInstructions,
  StreamSieve,
  SseTranslator,
  extractImages,
  extractAttachments,
  estimateTokens,
  sanitizeLeaked,
  formatToolCallsForPrompt,
} from "./lib.mjs";
import { DeepSeekWebPlugin } from "./index.mjs";

let passed = 0;
function ok(label, fn) {
  try {
    fn();
    passed++;
    console.log(`  ✓ ${label}`);
  } catch (e) {
    console.error(`  ✗ ${label}\n    ${e?.message}`);
    process.exitCode = 1;
  }
}
async function okAsync(label, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ✓ ${label}`);
  } catch (e) {
    console.error(`  ✗ ${label}\n    ${e?.message}`);
    process.exitCode = 1;
  }
}

// ---------- buildPrompt ----------
console.log("buildPrompt");
ok("单条用户消息原样透传", () => {
  assert.equal(buildPrompt([{ role: "user", content: "你好" }], null), "你好");
});
ok("系统提示走包装格式（保证会话复用的 record 前缀一致）", () => {
  assert.equal(
    buildPrompt([{ role: "system", content: "S" }, { role: "user", content: "U" }], null),
    "[系统设定]\nS\n\n[对话记录]\n用户：U",
  );
});
ok("多轮拍平成对话记录+指令", () => {
  const p = buildPrompt(
    [{ role: "user", content: "A" }, { role: "assistant", content: "B" }, { role: "user", content: "C" }],
    null,
  );
  assert.ok(p.includes("用户：A") && p.includes("助手：B") && p.includes("用户：C"));
  assert.ok(p.includes("请以助手的身份，直接回复上面对话记录中最后一条用户消息"));
});
ok("图片内容替换为编号占位", () => {
  const p = buildPrompt(
    [{ role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,xx" } }, { type: "text", text: "图里是啥" }] }],
    null,
  );
  assert.ok(p.includes("[图片 #1]") && p.includes("图里是啥"));
});

// ---------- 上下文裁剪与增量拆分 ----------
console.log("buildPromptParts");
ok("超预算裁剪：丢最老轮次、保护最后一条用户消息", () => {
  const msgs = [];
  for (const q of ["第一问", "第二问", "第三问", "最后的第四问"]) {
    msgs.push({ role: "user", content: q }, { role: "assistant", content: `答：${q}` });
  }
  const { record } = buildPromptParts(msgs, null, { maxContextTokens: 40 });
  assert.ok(!record.includes("第一问"), "最老轮次应被裁掉");
  assert.ok(record.includes("最后的第四问"), "最后一条用户消息必须保留");
});
ok("增量拆分：第二轮 record 是第一轮的延伸", () => {
  const p1 = buildPromptParts([{ role: "user", content: "Q1" }, { role: "assistant", content: "A1" }, { role: "user", content: "Q2" }], null);
  const p2 = buildPromptParts(
    [
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
      { role: "user", content: "Q2" },
      { role: "assistant", content: "A2" },
      { role: "user", content: "Q3" },
    ],
    null,
  );
  assert.ok(p2.record.startsWith(p1.record), "record 应单调延伸");
  assert.equal(p2.record.slice(p1.record.length), "\n助手：A2\n用户：Q3");
});
ok("tool result 截断选项", () => {
  const { record } = buildPromptParts(
    [
      { role: "user", content: "跑一下" },
      { role: "assistant", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } }] },
      { role: "tool", tool_call_id: "c1", content: "x".repeat(500) },
    ],
    [{ type: "function", function: { name: "bash", description: "d", parameters: { type: "object" } } }],
    { toolResultLimit: 100 },
  );
  assert.ok(record.includes("…（截断）") && !record.includes("x".repeat(200)));
});

// ---------- estimateTokens / sanitizeLeaked ----------
ok("token 估算：CJK 1 字 1 token、ASCII 4 字符 1 token", () => {
  assert.equal(estimateTokens("你好"), 2);
  assert.equal(estimateTokens("abcdefgh"), 2);
  assert.ok(estimateTokens("你好 world") >= 3);
});
ok("泄漏清理：原生 token 与引用标记", () => {
  assert.equal(sanitizeLeaked("a<｜Tool｜>b<｜begin▁of▁sentence｜>c[citation:3]d"), "abcd");
});

// ---------- DSML 工具协议 ----------
console.log("DSML 工具协议");
const TOOLS = [
  {
    type: "function",
    function: {
      name: "bash",
      description: "执行命令",
      parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] },
    },
  },
];
ok("工具模式注入 DSML 模板与工具清单", () => {
  const p = buildPrompt([{ role: "user", content: "列出文件" }], TOOLS);
  assert.ok(p.includes("<|DSML|tool_calls>") && p.includes('parameter name="参数名"'));
  assert.ok(p.includes("  - bash：执行命令"));
  assert.ok(p.includes("严禁编造工具的返回结果"));
  assert.ok(!p.includes("参数 JSON Schema"), "DSML 模式不带 JSON Schema");
});
ok("toolChoice required 追加强制指令", () => {
  const instr = buildToolInstructions(TOOLS, "required");
  assert.ok(instr.includes("本轮必须至少调用一个工具"));
});
ok("历史 tool_calls 回写成 DSML 块", () => {
  const p = buildPrompt(
    [
      { role: "user", content: "列出文件" },
      { role: "assistant", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } }] },
      { role: "tool", tool_call_id: "c1", content: "a.txt" },
    ],
    TOOLS,
  );
  assert.ok(p.includes('<|DSML|invoke name="bash">'));
  assert.ok(p.includes("<![CDATA[ls]]]") || p.includes("<![CDATA[ls]]>"));
  assert.ok(p.includes("[工具 bash 的返回结果]"));
});
ok("formatToolCallsForPrompt：CDATA 内引号/换行零转义（渲染→解析回环）", () => {
  const content = 'line1\n"quoted" }]| <|invoke';
  const block = formatToolCallsForPrompt([
    { function: { name: "write", arguments: JSON.stringify({ path: "a.txt", content }) } },
  ]);
  const { text, calls } = collect(feedAll(new StreamSieve({ toolNames: ["write"] }), block, 7));
  assert.equal(text, "");
  assert.equal(calls.length, 1);
  assert.deepEqual(JSON.parse(calls[0].arguments), { path: "a.txt", content });
});

// ---------- StreamSieve ----------
console.log("StreamSieve");
function feedAll(sieve, text, chunk = 3) {
  const events = [];
  for (let i = 0; i < text.length; i += chunk) events.push(...sieve.feed(text.slice(i, i + chunk)));
  events.push(...sieve.flush());
  return events;
}
function collect(events) {
  let text = "";
  const calls = [];
  for (const e of events) {
    if (e.type === "text") text += e.text;
    else calls.push(...e.calls);
  }
  return { text, calls };
}
const DSML_BLOCK =
  '<|DSML|tool_calls>\n  <|DSML|invoke name="bash">\n    <|DSML|parameter name="command"><![CDATA[ls -la && echo "a|b"]]></|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>';
ok("完整 DSML 块解析（CDATA 特殊字符）", () => {
  const { text, calls } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), `前言${DSML_BLOCK}后记`));
  assert.equal(text, "前言后记");
  assert.equal(calls.length, 1);
  assert.deepEqual(JSON.parse(calls[0].arguments), { command: 'ls -la && echo "a|b"' });
});
ok("逐字符分包零泄漏", () => {
  const { text, calls } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), DSML_BLOCK, 1));
  assert.equal(text, "");
  assert.equal(calls.length, 1);
  assert.deepEqual(JSON.parse(calls[0].arguments), { command: 'ls -la && echo "a|b"' });
});
ok("连续多块夹正文", () => {
  const two = `A${DSML_BLOCK}B${DSML_BLOCK.replace("bash", "bash")}C`;
  const { text, calls } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), two));
  assert.equal(text, "ABC");
  assert.equal(calls.length, 2);
});
ok("兼容旧 <tool_call> JSON 块", () => {
  const legacy = '前言<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>后记';
  const { text, calls } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), legacy));
  assert.equal(text, "前言后记");
  assert.deepEqual(JSON.parse(calls[0].arguments), { command: "ls" });
});
ok("loose JSON 修复（未引号 key/尾逗号，非内容键）", () => {
  const block = '<|DSML|tool_calls>\n  <|DSML|invoke name="t">\n    <|DSML|parameter name="payload"><![CDATA[{a: 1,}]]></|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>';
  const { calls } = collect(feedAll(new StreamSieve({ toolNames: ["t"] }), block));
  assert.deepEqual(JSON.parse(calls[0].arguments), { payload: { a: 1 } });
});
ok("同名参数合并成数组", () => {
  const block =
    '<|DSML|tool_calls>\n  <|DSML|invoke name="t">\n    <|DSML|parameter name="x"><![CDATA[1]]></|DSML|parameter>\n    <|DSML|parameter name="x"><![CDATA[2]]></|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>';
  const { calls } = collect(feedAll(new StreamSieve({ toolNames: ["t"] }), block));
  assert.deepEqual(JSON.parse(calls[0].arguments), { x: [1, 2] });
});
ok("CamelCase 工具名归一化", () => {
  const { calls } = collect(feedAll(new StreamSieve({ toolNames: ["get_weather"] }), DSML_BLOCK.replace("bash", "getWeather")));
  assert.equal(calls[0].name, "get_weather");
});
ok("裸 invoke（丢 wrapper）也能解析", () => {
  const bare = '  <|DSML|invoke name="bash">\n    <|DSML|parameter name="command"><![CDATA[ls]]></|DSML|parameter>\n  </|DSML|invoke>\n</|DSML|tool_calls>';
  const { calls } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), bare));
  assert.equal(calls.length, 1);
});
ok("未闭合块降级为正文", () => {
  const { text, calls } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), `hello <|DSML|tool_calls><|DSML|invoke name="bash">`));
  assert.equal(calls.length, 0);
  assert.ok(text.startsWith("hello"), JSON.stringify(text));
});
ok("孤立 < 与 | 不误判", () => {
  const { text } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), "a < b | c | d"));
  assert.equal(text, "a < b | c | d");
});
ok("缓冲超限强制按正文放行", () => {
  const sieve = new StreamSieve({ toolNames: ["bash"], maxBuffer: 64 });
  const adversarial = "<|DSML|tool_calls>" + "x".repeat(200);
  const { text } = collect([...sieve.feed(adversarial), ...sieve.flush()]);
  assert.equal(text, adversarial);
});
ok("普通文本带 HTML 标签不受影响", () => {
  const { text } = collect(feedAll(new StreamSieve({ toolNames: ["bash"] }), "看 <b>加粗</b> 和 <invoke 没写完"));
  assert.equal(text, "看 <b>加粗</b> 和 <invoke 没写完");
});

// ---------- extractImages / extractAttachments ----------
console.log("extractImages / extractAttachments");
ok("data URL 提取 mime 与 base64", () => {
  const imgs = extractImages([{ role: "user", content: [{ type: "image_url", image_url: { url: "data:image/jpeg;base64,QUJD" } }] }]);
  assert.equal(imgs.length, 1);
  assert.equal(imgs[0].mime, "image/jpeg");
  assert.equal(imgs[0].base64, "QUJD");
});
ok("相同 data URL 去重、超过上限截断", () => {
  const dup = [{ role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }] }];
  assert.equal(extractImages([...dup, ...dup]).length, 1);
  const many = Array.from({ length: 10 }, (_, i) => ({ role: "user", content: [{ type: "image_url", image_url: { url: `data:image/png;base64,${i}` } }] }));
  assert.equal(extractImages(many).length, 8);
});
ok("file part 提取附件", () => {
  const fs = extractAttachments([{ role: "user", content: [{ type: "file", filename: "notes.txt", file_data: "data:text/plain;base64,SGk=" }] }]);
  assert.equal(fs.length, 1);
  assert.equal(fs[0].mime, "text/plain");
  assert.equal(fs[0].name, "notes.txt");
});

// ---------- SseTranslator（含 hint / ready）----------
console.log("SseTranslator");
const SSE_OK = [
  "event: ready",
  'data: {"response_message_id": 101}',
  "",
  'data: {"v": {"response": {"fragments": [{"type": "RESPONSE", "content": ""}], "accumulated_token_usage": 123}}}',
  "",
  'data: {"p": "response/fragments", "o": "APPEND", "v": [{"type": "THINK", "content": "思考"}]}',
  "",
  'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "更多思考"}',
  "",
  'data: {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "回答"}]}',
  "",
  'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "第一段"}',
  "",
  'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "第二段"}',
  "",
  'data: {"p": "response/status", "o": "SET", "v": "FINISHED"}',
  "",
  "event: close",
  "data: ",
  "",
].join("\n");
ok("THINK→reasoning、RESPONSE→content、结束与用量、ready/hint 捕获", () => {
  const tr = new SseTranslator();
  let reasoning = "", content = "";
  for (let i = 0; i < SSE_OK.length; i += 3) {
    for (const piece of tr.feed(SSE_OK.slice(i, i + 3))) {
      if (piece.type === "reasoning") reasoning += piece.text;
      else content += piece.text;
    }
    if (tr.finished) break;
  }
  assert.equal(reasoning, "思考更多思考");
  assert.equal(content, "回答第一段第二段");
  assert.equal(tr.finished, true);
  assert.equal(tr.completionTokens, 123);
  assert.equal(tr.responseMessageId, 101);
  assert.equal(tr.hintError, null);
});
ok("hint 限流事件被捕获", () => {
  const tr = new SseTranslator();
  tr.feed('event: hint\ndata: {"type":"error","finish_reason":"rate_limit_reached","content":"消息发送过于频繁，请稍后重试"}\n\n');
  assert.equal(tr.hintError.finishReason, "rate_limit_reached");
  assert.ok(/频繁/.test(tr.hintError.content));
});
ok("toast 错误事件被捕获", () => {
  const tr = new SseTranslator();
  tr.feed('event: toast\ndata: {"type":"error","content":"客户端版本过低"}\n\n');
  assert.equal(tr.hintError.finishReason, "toast_error");
  assert.ok(tr.hintError.content.includes("客户端"));
});

// ---------- 插件行为（桩 fetch）----------
console.log("插件行为（桩 fetch）");

const REAL_CHALLENGE = {
  algorithm: "DeepSeekHashV1",
  challenge: "ea415cd70d0c9dabf1893bac23eacfe972349d254499dde39a3110d0badc18c0",
  salt: "4afc5c7d56f0ff06e1bf",
  signature: "4043caabdec7c7c35f1cc181143089d51969ddf3e578a20b0afcd5bd01269e43",
  difficulty: 144000,
  expire_at: 1793414496131,
  expire_after: 300000,
  target_path: "/api/v0/chat/completion",
};

function sseLine(obj) {
  return `data: ${JSON.stringify(obj)}\n\n`;
}
const COMPLETION_SSE_OK =
  'event: ready\ndata: {"response_message_id": 101}\n\n' +
  sseLine({ v: { response: { fragments: [{ type: "RESPONSE", content: "" }], accumulated_token_usage: 123 } } }) +
  sseLine({ p: "response/fragments/-1/content", o: "APPEND", v: "回答" }) +
  'data: {"p": "response/status", "o": "SET", "v": "FINISHED"}\n\nevent: close\ndata: \n\n';
const HINT_RATE_LIMIT =
  'event: hint\ndata: {"type":"error","finish_reason":"rate_limit_reached","content":"消息发送过于频繁，请稍后重试"}\n\nevent: close\ndata: \n\n';

let tokenSeq = 0;

async function newLoader(opts = {}) {
  const plugin = await DeepSeekWebPlugin(null, { retryAttempts: 2, retryBackoffMs: 5, ...opts });
  return plugin.auth.loader(async () => ({ type: "api", key: `tok-${++tokenSeq}` }));
}

function stubUpstream({ completions, onCreate, onDelete, capture } = {}) {
  // completions: 数组或函数(i => Response)
  let i = 0;
  globalThis.fetch = async (url, init) => {
    const u = String(url);
    if (u.includes("/chat_session/create")) {
      onCreate?.();
      return new Response('{"code":0,"data":{"biz_data":{"chat_session":{"id":"s1"}}}}', { status: 200 });
    }
    if (u.includes("/chat/create_pow_challenge")) {
      return new Response(JSON.stringify({ code: 0, data: { biz_data: { challenge: { ...REAL_CHALLENGE } } } }), { status: 200 });
    }
    if (u.includes("/chat/completion")) {
      capture?.(init);
      const resp = typeof completions === "function" ? completions(i++) : completions[Math.min(i++, completions.length - 1)];
      return resp;
    }
    if (u.includes("/chat_session/delete")) {
      onDelete?.();
      return new Response('{"code":0}', { status: 200 });
    }
    throw new Error(`stub: 未预期的请求 ${u}`);
  };
}
const RESP_SSE = (body) => new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });

await okAsync("正常回答（非流式）→ content + total_tokens + usage", async () => {
  stubUpstream({ completions: () => RESP_SSE(COMPLETION_SSE_OK) });
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 200);
  const j = await r.json();
  assert.equal(j.choices[0].message.content, "回答");
  assert.equal(j.choices[0].finish_reason, "stop");
  assert.equal(j.usage.total_tokens, 123);
});
await okAsync("DSML 工具调用 → tool_calls 结构", async () => {
  const block = DSML_BLOCK.replace('name="bash"', 'name="bash"');
  const body =
    'event: close\ndata: \n\n'.length > 0
      ? sseLine({ v: { response: { fragments: [{ type: "RESPONSE", content: "" }] } } }) +
        sseLine({ p: "response/fragments/-1/content", o: "APPEND", v: block }) +
        'data: {"p": "response/status", "o": "SET", "v": "FINISHED"}\n\nevent: close\ndata: \n\n'
      : "";
  stubUpstream({ completions: () => RESP_SSE(body) });
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, {
    body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "列出文件" }], tools: TOOLS }),
  });
  const j = await r.json();
  assert.equal(j.choices[0].finish_reason, "tool_calls");
  assert.equal(j.choices[0].message.tool_calls[0].function.name, "bash");
  assert.deepEqual(JSON.parse(j.choices[0].message.tool_calls[0].function.arguments), { command: 'ls -la && echo "a|b"' });
});
await okAsync("空流（非流式）→ 502 显式错误", async () => {
  stubUpstream({ completions: () => RESP_SSE("") });
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 502);
  const j = await r.json();
  assert.match(j.error.message, /空回答/);
});
await okAsync("空流（流式）→ error 事件 + [DONE]", async () => {
  stubUpstream({ completions: () => RESP_SSE("event: close\n\ndata: \n\n") });
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: true, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 200);
  const body = await r.text();
  assert.ok(body.includes('"error"') && body.includes("[DONE]"));
});
await okAsync("限流 hint → 退避重试 → 成功", async () => {
  stubUpstream({ completions: (i) => RESP_SSE(i === 0 ? HINT_RATE_LIMIT : COMPLETION_SSE_OK) });
  const loader = await newLoader({ sessionReuse: false, rateLimitRetryDelays: [5] });
  let calls = 0;
  const orig = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (String(url).includes("/chat/completion")) calls++;
    return orig(url, init);
  };
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  globalThis.fetch = orig;
  assert.equal(r.status, 200);
  const j = await r.json();
  assert.equal(j.choices[0].message.content, "回答");
  assert.equal(calls, 2, "限流后应重试一次");
});
await okAsync("禁言（JSON 200 biz_code=5）→ 429 不重试", async () => {
  stubUpstream({ completions: () => new Response('{"code":0,"data":{"biz_code":5,"biz_msg":"user is muted"}}', { status: 200 }) });
  const loader = await newLoader({ sessionReuse: false });
  let calls = 0;
  const orig = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (String(url).includes("/chat/completion")) calls++;
    return orig(url, init);
  };
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  globalThis.fetch = orig;
  assert.equal(r.status, 429);
  const j = await r.json();
  assert.match(j.error.message, /禁言/);
  assert.equal(calls, 1, "禁言不应重试");
});
await okAsync("空流重试：第一次空、第二次成功（非流式）", async () => {
  stubUpstream({ completions: (i) => RESP_SSE(i === 0 ? "" : COMPLETION_SSE_OK) });
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 200);
  const j = await r.json();
  assert.equal(j.choices[0].message.content, "回答");
});
await okAsync("空流重试：流式模式重试对客户端透明", async () => {
  stubUpstream({ completions: (i) => RESP_SSE(i === 0 ? "" : COMPLETION_SSE_OK) });
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: true, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 200);
  const body = await r.text();
  const content = body
    .split("\n")
    .filter((l) => l.startsWith("data: ") && !l.includes("[DONE]"))
    .map((l) => {
      try {
        return JSON.parse(l.slice(6));
      } catch {
        return null;
      }
    })
    .filter((c) => c?.choices?.[0]?.delta?.content)
    .map((c) => c.choices[0].delta.content)
    .join("");
  assert.equal(content, "回答");
  assert.ok(body.includes("[DONE]") && !body.includes('"error"'));
});
await okAsync("会话复用：第二轮发增量 prompt + parent_message_id，会话不删", async () => {
  const bodies = [];
  let creates = 0;
  let deletes = 0;
  stubUpstream({
    completions: () => RESP_SSE(COMPLETION_SSE_OK),
    onCreate: () => creates++,
    onDelete: () => deletes++,
    capture: (init) => bodies.push(JSON.parse(init.body)),
  });
  const loader = await newLoader({ sessionReuse: true });
  const msgs1 = [{ role: "user", content: "Q1" }, { role: "assistant", content: "A1" }, { role: "user", content: "Q2" }];
  await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: msgs1 }) });
  const msgs2 = [...msgs1, { role: "assistant", content: "A2" }, { role: "user", content: "Q3" }];
  await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: msgs2 }) });
  assert.equal(bodies.length, 2);
  assert.equal(bodies[0].chat_session_id, "s1");
  assert.equal(bodies[1].chat_session_id, "s1", "第二轮应复用同一会话");
  assert.equal(bodies[1].parent_message_id, 101, "parent_message_id 应来自 ready 帧");
  assert.equal(
    bodies[1].prompt,
    "助手：A2\n用户：Q3\n\n请以助手的身份，直接回复上面对话记录中最后一条用户消息。不要复述记录本身。",
    "只发增量尾部+规则",
  );
  assert.equal(creates, 1, "只建一次会话");
  assert.equal(deletes, 0, "复用模式不删会话");
});
await okAsync("sessionReuse: false → 每轮新建并删除会话", async () => {
  const bodies = [];
  let creates = 0;
  let deletes = 0;
  stubUpstream({
    completions: () => RESP_SSE(COMPLETION_SSE_OK),
    onCreate: () => creates++,
    onDelete: () => deletes++,
    capture: (init) => bodies.push(JSON.parse(init.body)),
  });
  const loader = await newLoader({ sessionReuse: false });
  const msgs = [{ role: "user", content: "hi" }];
  await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: msgs }) });
  await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: msgs }) });
  assert.equal(creates, 2);
  assert.equal(deletes, 2);
  assert.equal(bodies[1].parent_message_id, null);
});
await okAsync("40003/INVALID_TOKEN → 401 + 登录标记", async () => {
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/chat_session/create")) return new Response('{"code":40003,"msg":"INVALID_TOKEN","data":null}', { status: 200 });
    throw new Error(`stub: 未预期的请求 ${u}`);
  };
  const loader = await newLoader({ sessionReuse: false });
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 401);
  assert.equal(r.headers.get("x-magpie-sign-in"), "expired");
});
await okAsync("采样参数透传（temperature/max_tokens）", async () => {
  const bodies = [];
  stubUpstream({ completions: () => RESP_SSE(COMPLETION_SSE_OK), capture: (init) => bodies.push(JSON.parse(init.body)) });
  const loader = await newLoader({ sessionReuse: false });
  await loader.fetch(null, {
    body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }], temperature: 0.5, top_p: 0.9, max_tokens: 777 }),
  });
  assert.equal(bodies[0].temperature, 0.5);
  assert.equal(bodies[0].top_p, 0.9);
  assert.equal(bodies[0].max_tokens, 777);
});
await okAsync("search 模型变体 → search_enabled", async () => {
  const bodies = [];
  stubUpstream({ completions: () => RESP_SSE(COMPLETION_SSE_OK), capture: (init) => bodies.push(JSON.parse(init.body)) });
  const loader = await newLoader({ sessionReuse: false });
  await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-reasoner-search", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(bodies[0].search_enabled, true);
  assert.equal(bodies[0].thinking_enabled, true);
});


console.log(`\n${passed} 项通过${process.exitCode ? "（有失败）" : ""}`);
