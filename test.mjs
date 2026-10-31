// 单元测试：bun test.mjs（或 node --experimental-vm-modules，推荐 bun）
// 覆盖：messages 拍平、<tool_call> 流式提取、SSE 转写、空回答显式报错、业务码鉴权映射。
import assert from "node:assert/strict";
import { buildPrompt, SseTranslator, ToolCallStream } from "./lib.mjs";
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
ok("系统提示拼在最前", () => {
  assert.equal(
    buildPrompt([{ role: "system", content: "S" }, { role: "user", content: "U" }], null),
    "S\n\nU",
  );
});
ok("多轮拍平成对话记录", () => {
  const p = buildPrompt(
    [{ role: "user", content: "A" }, { role: "assistant", content: "B" }, { role: "user", content: "C" }],
    null,
  );
  assert.ok(p.includes("用户：A") && p.includes("助手：B") && p.includes("用户：C"));
  assert.ok(p.includes("直接回复上面对话记录中最后一条用户消息"));
});
ok("图片内容替换为占位说明", () => {
  const p = buildPrompt(
    [{ role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,xx" } }, { type: "text", text: "图里是啥" }] }],
    null,
  );
  assert.ok(p.includes("网页版暂不支持") && p.includes("图里是啥"));
});

// ---------- 工具模式 ----------
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
ok("工具模式包含工具定义与回复规则", () => {
  const p = buildPrompt([{ role: "user", content: "列出文件" }], TOOLS);
  assert.ok(p.includes("### bash") && p.includes("执行命令") && p.includes("参数 JSON Schema"));
  assert.ok(p.includes("<tool_call>") && p.includes("严禁编造工具的返回结果"));
});
ok("历史里的工具调用渲染成协议块，工具结果映射回用户侧", () => {
  const p = buildPrompt(
    [
      { role: "user", content: "列出文件" },
      { role: "assistant", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } }] },
      { role: "tool", tool_call_id: "c1", content: "a.txt" },
    ],
    TOOLS,
  );
  assert.ok(p.includes('{"name": "bash", "arguments": {"command":"ls"}}'));
  assert.ok(p.includes("[工具 bash 的返回结果]"));
});

// ---------- ToolCallStream ----------
console.log("ToolCallStream");
ok("完整块解析成调用", () => {
  const tc = new ToolCallStream();
  const a = tc.feed('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>');
  const b = tc.end();
  assert.deepEqual([...a.calls, ...b.calls].map((c) => c.name), ["bash"]);
  assert.deepEqual([...a.calls, ...b.calls].map((c) => c.arguments), ['{"command":"ls"}']);
  assert.equal((a.text + b.text).trim(), "");
});
ok("逐字符分包结果一致", () => {
  const tc = new ToolCallStream();
  const src = '前言<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>后记';
  let text = "";
  const calls = [];
  for (const ch of src) {
    const o = tc.feed(ch);
    text += o.text;
    calls.push(...o.calls);
  }
  const tail = tc.end();
  text += tail.text;
  calls.push(...tail.calls);
  assert.equal(text, "前言后记");
  assert.equal(calls.length, 1);
  assert.equal(calls[0].name, "bash");
});
ok("剥掉包裹调用块的围栏", () => {
  const tc = new ToolCallStream();
  const o = tc.feed('```json\n<tool_call>\n{"name": "bash", "arguments": {}}\n</tool_call>\n```');
  const t = tc.end();
  assert.equal((o.text + t.text).replace(/`\n?/g, ""), "");
  assert.equal([...o.calls, ...t.calls].length, 1);
});
ok("JSON 坏了的块原文放行", () => {
  const tc = new ToolCallStream();
  const a = tc.feed("<tool_call>\n{oops}\n</tool_call>");
  const b = tc.end();
  assert.equal(a.text + b.text, "<tool_call>\n{oops}\n</tool_call>");
  assert.equal([...a.calls, ...b.calls].length, 0);
});
ok("多个块与正文混合", () => {
  const tc = new ToolCallStream();
  const a = tc.feed('A<tool_call>{"name":"f1","arguments":{}}</tool_call>B<tool_call>{"name":"f2","arguments":{}}</tool_call>C');
  const b = tc.end();
  assert.equal(a.text + b.text, "ABC");
  assert.deepEqual([...a.calls, ...b.calls].map((c) => c.name), ["f1", "f2"]);
});

// ---------- SseTranslator ----------
console.log("SseTranslator");
const SSE_OK = [
  'event: update_session',
  'data: {"v": ""}',
  '',
  'data: {"v": {"response": {"fragments": [{"type": "RESPONSE", "content": ""}], "accumulated_token_usage": 123}}}',
  '',
  'data: {"p": "response/fragments", "o": "APPEND", "v": [{"type": "THINK", "content": "思考"}]}',
  '',
  'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "更多思考"}',
  '',
  'data: {"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "回答"}]}',
  '',
  'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "第一段"}',
  '',
  'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "第二段"}',
  '',
  'data: {"p": "response/status", "o": "SET", "v": "FINISHED"}',
  '',
  'event: close',
  'data: ',
  '',
].join("\n");
ok("THINK→reasoning、RESPONSE→content、结束与用量", () => {
  const tr = new SseTranslator();
  // 任意分包：拆成 3 字符一片
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
});

// ---------- 插件级行为（桩 fetch）----------
console.log("插件行为（桩 fetch）");

const realFetch = globalThis.fetch;
function stubFetch(completionBody, { completionStatus = 200, contentType = "text/event-stream" } = {}) {
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/chat_session/create")) {
      return new Response('{"code":0,"data":{"biz_data":{"chat_session":{"id":"sess1"}}}}', { status: 200 });
    }
    if (u.includes("/chat/create_pow_challenge")) {
      // 真实形态的挑战样本（服务器只在校验头时用它，桩里 completion 端点并不校验）
      return new Response(
        JSON.stringify({
          code: 0,
          data: {
            biz_data: {
              challenge: {
                algorithm: "DeepSeekHashV1",
                challenge: "ea415cd70d0c9dabf1893bac23eacfe972349d254499dde39a3110d0badc18c0",
                salt: "4afc5c7d56f0ff06e1bf",
                signature: "4043caabdec7c7c35f1cc181143089d51969ddf3e578a20b0afcd5bd01269e43",
                difficulty: 144000,
                expire_at: 1793414496131,
                expire_after: 300000,
                target_path: "/api/v0/chat/completion",
              },
            },
          },
        }),
        { status: 200 },
      );
    }
    if (u.includes("/chat/completion")) {
      return new Response(completionBody, { status: completionStatus, headers: { "content-type": contentType } });
    }
    if (u.includes("/chat_session/delete")) return new Response('{"code":0}', { status: 200 });
    throw new Error(`stub: 未预期的请求 ${u}`);
  };
}
const plugin = await DeepSeekWebPlugin(null, {});
const loader = await plugin.auth.loader(async () => ({ type: "api", key: "tok" }));

await okAsync("空流（非流式）→ 502 显式错误", async () => {
  stubFetch("");
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 502);
  const j = await r.json();
  assert.match(j.error.message, /空回答/);
});
await okAsync("空流（流式）→ error 事件 + [DONE]", async () => {
  stubFetch("event: close\n\ndata: \n\n");
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: true, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 200);
  const body = await r.text();
  assert.ok(body.includes('"error"') && body.includes("[DONE]"));
});
await okAsync("正常回答（非流式）→ content + total_tokens", async () => {
  stubFetch(SSE_OK);
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 200);
  const j = await r.json();
  assert.equal(j.choices[0].message.content, "回答第一段第二段");
  assert.equal(j.choices[0].finish_reason, "stop");
  assert.equal(j.usage.total_tokens, 123);
  assert.equal(j.usage.completion_tokens, 0);
});
await okAsync("回答里的调用块 → tool_calls 结构", async () => {
  const sse = [
    'data: {"v": {"response": {"fragments": [{"type": "RESPONSE", "content": ""}]}}}',
    "",
    'data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "<tool_call>\\n{\\"name\\": \\"bash\\", \\"arguments\\": {\\"command\\": \\"ls\\"}}\\n</tool_call>"}',
    "",
    'data: {"p": "response/status", "o": "SET", "v": "FINISHED"}',
    "",
    "event: close",
    "data: ",
    "",
  ].join("\n");
  stubFetch(sse);
  const r = await loader.fetch(null, {
    body: JSON.stringify({ model: "deepseek-chat", stream: false, messages: [{ role: "user", content: "列出文件" }], tools: TOOLS }),
  });
  assert.equal(r.status, 200);
  const j = await r.json();
  assert.equal(j.choices[0].finish_reason, "tool_calls");
  assert.equal(j.choices[0].message.tool_calls[0].function.name, "bash");
  assert.equal(j.choices[0].message.tool_calls[0].function.arguments, '{"command":"ls"}');
});
await okAsync("40003/INVALID_TOKEN → 401 + 登录标记", async () => {
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/chat_session/create")) return new Response('{"code":40003,"msg":"INVALID_TOKEN","data":null}', { status: 200 });
    throw new Error(`stub: 未预期的请求 ${u}`);
  };
  const r = await loader.fetch(null, { body: JSON.stringify({ model: "deepseek-chat", messages: [{ role: "user", content: "hi" }] }) });
  assert.equal(r.status, 401);
  assert.equal(r.headers.get("x-magpie-sign-in"), "expired");
});

globalThis.fetch = realFetch;
console.log(`\n${passed} 项通过${process.exitCode ? "（有失败）" : ""}`);
