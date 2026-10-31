// 纯函数部分：messages 拍平（含 DSML 工具协议与上下文裁剪）、
// DeepSeek SSE → OpenAI chunk 转换（含 hint/toast/ready 事件）、流式工具调用筛分。
//
// 工具协议采用 DSML（<|DSML|invoke ...><![CDATA[…]]>，与 DeepSeek 原生特殊 token 同风格），
// 同时兼容旧的 <tool_call> JSON 块——参考 freeapi / web2api 两个逆向项目的实现。

function contentToText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  let imgIdx = 0;
  for (const p of content) {
    if (p?.type === "text" && p.text) parts.push(p.text);
    else if (p?.type === "image" || p?.type === "image_url") parts.push(`[图片 #${++imgIdx}]`);
    else if (typeof p?.text === "string") parts.push(p.text);
  }
  return parts.join("");
}

// 提取 messages 里的图片，供上传到网页版后用 ref_file_ids 引用。
// 支持 data URL（base64 直传）与 http(s) URL（请求时下载）。
// 返回 [{ mime, base64?, url?, name }]；identical data URL 去重，上限 maxImages。
export function extractImages(messages, maxImages = 8) {
  const seen = new Set();
  const out = [];
  for (const m of messages ?? []) {
    if (!Array.isArray(m?.content)) continue;
    for (const p of m.content) {
      if (p?.type !== "image" && p?.type !== "image_url") continue;
      const url = p?.image_url?.url ?? p?.url ?? (typeof p?.image === "string" ? p.image : null);
      if (!url || typeof url !== "string") continue;
      let item = null;
      const dm = /^data:([^;,]+);base64,(.+)$/s.exec(url);
      if (dm) {
        item = { mime: dm[1], base64: dm[2] };
      } else if (/^https?:\/\//.test(url)) {
        item = { mime: null, url };
      }
      if (!item) continue;
      const key = item.base64 ? `b:${item.base64.slice(0, 4096)}:${item.base64.length}` : `u:${url}`;
      if (seen.has(key)) continue;
      seen.add(key);
      item.name = `image-${out.length + 1}.${extFor(item.mime)}`;
      out.push(item);
      if (out.length >= maxImages) return out;
    }
  }
  return out;
}

function extFor(mime) {
  const map = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif" };
  return map[mime ?? ""] ?? "png";
}

// 提取非图片附件（约定 content part：{ type: "file", filename?, file_data: "data:mime;base64,…" }）。
// OpenAI chat completions 没有标准的文件 part，这是插件扩展；纯文本内容照旧走 prompt。
export function extractAttachments(messages, maxFiles = 8) {
  const out = [];
  for (const m of messages ?? []) {
    if (!Array.isArray(m?.content)) continue;
    for (const p of m.content) {
      if (p?.type !== "file") continue;
      const dm = /^data:([^;,]+);base64,(.+)$/s.exec(typeof p?.file_data === "string" ? p.file_data : "");
      if (!dm) continue;
      const mime = dm[1];
      out.push({
        mime,
        base64: dm[2],
        name: (typeof p?.filename === "string" && p.filename) || `file-${out.length + 1}.${extFor(mime)}`,
      });
      if (out.length >= maxFiles) return out;
    }
  }
  return out;
}

// ---- token 估算（保守：CJK 每字 1 token，其余每 4 字符 1 token，空白不计）----
const CJK_RE = /[\u3000-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFF00-\uFFEF]/g;
export function estimateTokens(text) {
  if (!text) return 0;
  const cjk = (text.match(CJK_RE) ?? []).length;
  const other = text.replace(CJK_RE, "").replace(/\s/g, "").length;
  return cjk + Math.max(other > 0 ? 1 : 0, Math.ceil(other / 4));
}

// ---- 泄漏清理：模型偶尔吐出原生特殊 token / 残留协议标签 / 引用标记 ----
const LEAK_RE =
  /<\/?\s*[|｜]*\s*(?:dsml\s*[|｜]*)*(?:tool_calls|tool_call|calls|invoke|parameter)\b[^>]{0,120}>|<\s*[|｜]\s*[a-zA-Z0-9▁_]{1,40}\s*[|｜]\s*>|\[citation:\d+\]/g;
export function sanitizeLeaked(text) {
  if (!text) return text;
  return text.replace(LEAK_RE, "");
}

// ---- DSML 工具协议 ----
//
// 模板格式（参考 ds2api / freeapi / web2api）：
//   <|DSML|tool_calls>
//     <|DSML|invoke name="工具名">
//       <|DSML|parameter name="参数名"><![CDATA[值]]></|DSML|parameter>
//     </|DSML|invoke>
//   </|DSML|tool_calls>
// 字符串值包 CDATA（引号/换行/代码零转义）；数字/布尔/null 纯文本。
// 分隔符 | 与全宽 ｜、大小写、重复管道都容忍。

function cdataEscape(s) {
  return String(s).replaceAll("]]>", "]]]]><![CDATA[>");
}

function xmlAttr(s) {
  return String(s).replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function htmlUnescape(s) {
  return String(s)
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replaceAll("&#39;", "'")
    .replaceAll("&amp;", "&")
    .replaceAll(/<br\s*\/?>/gi, "\n");
}

// 历史里 assistant 的 tool_calls 回写成 DSML 块（模型上轮见过自己输出格式的自洽性）
export function formatToolCallsForPrompt(toolCalls) {
  const invokes = (toolCalls ?? []).map((c) => {
    const name = c.function?.name ?? "?";
    let obj = {};
    const raw = c.function?.arguments ?? "{}";
    try {
      obj = typeof raw === "string" ? JSON.parse(raw || "{}") : raw ?? {};
    } catch {
      obj = {};
    }
    if (obj === null || typeof obj !== "object" || Array.isArray(obj)) obj = { value: obj };
    const params = Object.entries(obj)
      .map(([k, v]) => {
        // DSML 规则：字符串值必须 CDATA 包裹；数字/布尔/null 纯文本
        const val = typeof v === "string" ? `<![CDATA[${cdataEscape(v)}]]>` : v === undefined ? "null" : JSON.stringify(v);
        return `    <|DSML|parameter name="${xmlAttr(k)}">${val}</|DSML|parameter>`;
      })
      .join("\n");
    return `  <|DSML|invoke name="${xmlAttr(name)}">\n${params ? params + "\n" : ""}  </|DSML|invoke>`;
  });
  return `<|DSML|tool_calls>\n${invokes.join("\n")}\n</|DSML|tool_calls>`;
}

// 工具协议说明（注入 prompt 末尾；规则放最后指令遵循最强）
export function buildToolInstructions(tools, toolChoice) {
  const list = (tools ?? []).filter((t) => t?.type === "function" && t.function?.name);
  if (!list.length) return "";
  let extra = "";
  if (toolChoice === "required") extra = "本轮必须至少调用一个工具，不要直接给最终答复。";
  else if (toolChoice && typeof toolChoice === "object" && toolChoice?.function?.name)
    extra = `本轮必须调用工具 ${toolChoice.function.name}，不要调用其他工具，不要直接回答。`;
  else if (typeof toolChoice === "string" && !["auto", "none"].includes(toolChoice))
    extra = `本轮必须调用工具 ${toolChoice}，不要调用其他工具，不要直接回答。`;

  const toolLines = list
    .map((t) => {
      const desc = (t.function.description ?? "").trim();
      return `  - ${t.function.name}${desc ? `：${desc}` : ""}`;
    })
    .join("\n");

  return [
    "[可用工具]",
    "需要调用工具时，原样输出下面的调用块（第一行必须是 <|DSML|tool_calls>，不要放进代码块，前后不要附加任何解释文字；字符串参数一律用 CDATA 包裹）：",
    "",
    "<|DSML|tool_calls>",
    '  <|DSML|invoke name="工具名">',
    '    <|DSML|parameter name="参数名"><![CDATA[参数值]]></|DSML|parameter>',
    "  </|DSML|invoke>",
    "</|DSML|tool_calls>",
    "",
    "规则：",
    "- 工具名写在 invoke 的 name 属性；每个参数一个 parameter 节点，参数名写在 name 属性。",
    "- 字符串值必须用 <![CDATA[...]]> 包裹；数字、布尔、null 直接写纯文本。",
    '- 可以在同一个 <|DSML|tool_calls> 块里连续写多个 <|DSML|invoke>；调用块输出完必须立即停笔。',
    "- 严禁编造工具的返回结果：不要替“用户”写任何消息，不要自己写出“[工具…的返回结果]”，不要假装已经看到结果。真实结果由系统在下一轮附进对话记录。",
    "- 不需要调用工具时，直接写给用户看的答复正文，不要输出调用块。不要复述对话记录。",
    extra,
    "",
    "可用工具：",
    toolLines,
  ]
    .filter((l) => l !== undefined)
    .join("\n");
}

// ---- 工具调用块的流式筛分（StreamSieve）----
//
// 从正文流里实时剥出调用块（DSML 与旧 <tool_call> JSON 都认），容忍：
//   - 任意分包（含逐字符）：未完成的 "<|"/"|DS" 等前缀从最早可能处扣住
//   - markdown 围栏包裹、前后空白行
//   - 连续多个块、块间夹正文（suffix 迭代处理，不递归）
//   - 缓冲 1MiB 硬上限：超限强制按正文放行，防对抗性输出卡死
// 解析失败降级为正文，永不静默丢内容。

const START_CANDIDATES = [
  "<|dsml|tool_calls>",
  "<|dsml|invoke",
  "|dsml|tool_calls>",
  "|dsml|invoke",
  "<tool_calls>",
  "<tool_call>",
  "<invoke",
  "<calls>",
];
const MAX_TAG_LEN = Math.max(...START_CANDIDATES.map((t) => t.length));
const SIEVE_CAP = 1024 * 1024;

// 在缓冲里找最早的完整开标签（DSML 标记或普通 wrapper），返回 { index, kind } 或 null
function findStart(text) {
  const dsmlRe = /(?:<|(?<![|｜<]))[|｜]+\s*dsml\s*[|｜]+\s*(?:tool_calls|invoke)\b/gi;
  let best = -1;
  const consider = (i) => {
    if (i >= 0 && (best < 0 || i < best)) best = i;
  };
  let m = dsmlRe.exec(text);
  if (m) consider(m.index);
  const lower = text.toLowerCase();
  for (const t of ["<tool_calls>", "<tool_call>", "<calls>", "<invoke "]) {
    const i = lower.indexOf(t);
    if (i >= 0) consider(i);
  }
  // "<invoke>" 无空格变体
  const iv = lower.indexOf("<invoke>");
  if (iv >= 0) consider(iv);
  return best;
}

// 正文侧扣住策略：尾部如果可能是任何候选开标签的前缀（从最早的 '<'/'｜' 起），就扣住不发
function holdIndex(text) {
  if (text.length < 2) {
    // 单字符：'<' 或管道都可能是前缀
    return /[<|｜]/.test(text) ? 0 : -1;
  }
  const from = Math.max(0, text.length - MAX_TAG_LEN);
  const lower = text.toLowerCase();
  for (let i = from; i < text.length; i++) {
    const ch = text[i];
    if (ch !== "<" && ch !== "|" && ch !== "｜") continue;
    const tail = lower.slice(i);
    if (tail.length >= MAX_TAG_LEN) continue;
    for (const c of START_CANDIDATES) {
      if (c.startsWith(tail)) return i;
    }
  }
  return -1;
}

// 找捕获区里第一个完整块的结束偏移（CDATA 当单个 token，防止值里的字面标签截断结构）。
// 注意 ">" 必须显式匹配：可选的 ">" 会把未写完的闭合标签提前当成完整块。
function blockEnd(buf) {
  const re = /<\/?\s*[|｜]*\s*(?:dsml\s*[|｜]+\s*)?(?:tool_calls|tool_call|calls|invoke)\b[^>]*>|<!\[CDATA\[[\s\S]*?(?:\]\]>|$)/gi;
  let m;
  let openKind = null;
  while ((m = re.exec(buf))) {
    const tok = m[0];
    if (tok.startsWith("<!")) continue; // CDATA 原子跳过
    const lower = tok.toLowerCase();
    const isClose = lower.startsWith("</");
    const kind = /tool_calls|tool_call|calls|invoke/.exec(lower)?.[0] ?? "";
    const norm = kind === "tool_call" || kind === "calls" ? "tool_calls" : kind;
    if (!openKind) {
      if (!isClose && (norm === "tool_calls" || norm === "invoke")) openKind = norm;
      continue;
    }
    if (isClose && norm === openKind) return m.index + m[0].length;
  }
  return -1;
}

function repairLooseJson(s) {
  let out = s.replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)/g, '$1"$2"$3');
  out = out.replace(/,(\s*[}\]])/g, "$1");
  out = out.replace(/\\(?![\\/"bfnrtu])/g, "\\\\");
  return out;
}

const CDATA_STRING_KEYS = new Set([
  "content", "file_content", "text", "prompt", "query", "command", "cmd", "script", "code",
  "old_string", "new_string", "pattern", "path", "file_path", "result", "input", "value",
  "data", "message", "body", "output", "description", "url",
]);

function attrVal(attrs, name) {
  const m = new RegExp(`${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, "i").exec(attrs);
  return m ? (m[1] ?? m[2]) : null;
}

function resolveToolName(name, toolNames) {
  if (!name || !toolNames?.length) return name;
  if (toolNames.includes(name)) return name;
  const lower = name.toLowerCase();
  const hit = toolNames.find((t) => t.toLowerCase() === lower);
  if (hit) return hit;
  const snake = name.replace(/(?<=[a-z0-9])([A-Z])/g, "_$1").toLowerCase();
  return toolNames.includes(snake) ? snake : name;
}

function processParamValue(name, raw) {
  let v = raw.trim();
  let cdata = false;
  if (v.startsWith("<![CDATA[")) {
    const end = v.lastIndexOf("]]>");
    if (end >= 9) {
      v = v.slice(9, end);
      cdata = true;
    }
  }
  const t = v.trim();
  const key = (name ?? "").toLowerCase();
  if (cdata && CDATA_STRING_KEYS.has(key)) return htmlUnescape(v);
  if (/^(true|false)$/i.test(t)) return /^true/i.test(t);
  if (/^null$/i.test(t)) return null;
  if (/^-?\d+(\.\d+)?$/.test(t)) return Number(t);
  if (t.startsWith("{") || t.startsWith("[")) {
    try {
      return JSON.parse(t);
    } catch {}
    const rep = repairLooseJson(t);
    if (rep !== null) {
      try {
        return JSON.parse(rep);
      } catch {}
    }
  }
  return htmlUnescape(v);
}

// 解析一个完整调用块（DSML 优先，回退旧 <tool_call> JSON）
function parseToolBlock(block, toolNames, loose = false) {
  if (loose) {
    const openC = (block.match(/<!\[CDATA\[/g) ?? []).length;
    const closeC = (block.match(/\]\]>/g) ?? []).length;
    if (openC > closeC) block += "]]>";
  }
  const hasOpenWrapper = /<\s*[|｜]*\s*(?:dsml\s*[|｜]+\s*)?(?:tool_calls|tool_call|calls)\s*>/i.test(block);
  const hasInvoke = /<\s*[|｜]*\s*(?:dsml\s*[|｜]+\s*)?invoke\b/i.test(block);
  if (!hasOpenWrapper && hasInvoke) block = "<|DSML|tool_calls>\n" + block;

  const calls = [];
  // 每次调用新建正则（/g 的 lastIndex 会在多次 parseToolBlock 间泄漏）
  const bodyPat = '((?:<!\\[CDATA\\[[\\s\\S]*?\\]\\]>|(?!<\\s*/\\s*[|｜]*\\s*(?:dsml\\s*[|｜]+\\s*)?(?:tool_calls|tool_call|calls|invoke))[\\s\\S])*?)';
  const invokeRe = new RegExp(
    `<\\s*[|｜]*\\s*(?:dsml\\s*[|｜]+\\s*)?invoke\\s*([^>]*?)>${bodyPat}<\\s*/\\s*[|｜]*\\s*(?:dsml\\s*[|｜]+\\s*)?invoke\\s*>`,
    "gi",
  );
  let im;
  while ((im = invokeRe.exec(block))) {
    const name = attrVal(im[1] ?? "", "name") ?? "?";
    const args = {};
    const paramRe = new RegExp(
      `<\\s*[|｜]*\\s*(?:dsml\\s*[|｜]+\\s*)?parameter\\s*([^>]*?)>${bodyPat}<\\s*/\\s*[|｜]*\\s*(?:dsml\\s*[|｜]+\\s*)?parameter\\s*>`,
      "gi",
    );
    let pm;
    while ((pm = paramRe.exec(im[2]))) {
      const key = attrVal(pm[1] ?? "", "name") ?? "参数";
      const val = processParamValue(key, pm[2]);
      if (key in args) {
        args[key] = Array.isArray(args[key]) ? [...args[key], val] : [args[key], val];
      } else {
        args[key] = val;
      }
    }
    calls.push({
      name: resolveToolName(name, toolNames),
      arguments: JSON.stringify(args, (_k, v) => (v === undefined ? null : v)),
    });
  }
  if (calls.length) return calls;

  // 旧协议：<tool_call>{"name","arguments"}</tool_call>
  const legacy = /<tool_call>([\s\S]*?)<\/tool_call>/i.exec(block) ?? /<tool_calls>([\s\S]*?)<\/tool_calls>/i.exec(block);
  if (legacy) {
    let obj = null;
    const inner = legacy[1].trim();
    try {
      obj = JSON.parse(inner);
    } catch {}
    if (!obj) {
      const rep = repairLooseJson(inner);
      if (rep !== null) {
        try {
          obj = JSON.parse(rep);
        } catch {}
      }
    }
    if (obj && typeof obj === "object") {
      const arr = Array.isArray(obj) ? obj : [obj];
      for (const o of arr) {
        if (!o || typeof o.name !== "string" || !o.name) continue;
        let args = o.arguments ?? {};
        if (typeof args !== "string") {
          try {
            args = JSON.stringify(args);
          } catch {
            args = "{}";
          }
        }
        calls.push({ name: resolveToolName(o.name, toolNames), arguments: args });
      }
    }
  }
  return calls;
}

export class StreamSieve {
  constructor({ toolNames = [], maxBuffer = SIEVE_CAP } = {}) {
    this.toolNames = toolNames;
    this.max = Math.max(1024, maxBuffer);
    this.pending = "";
    this.capture = null; // { buf }
  }

  // 喂入正文增量，返回 [{ type: "text", text } | { type: "calls", calls }]
  feed(text) {
    const events = [];
    let buf = this.pending + text;
    this.pending = "";
    for (;;) {
      if (this.capture) {
        this.capture.buf += buf; // 新数据并入捕获区
        buf = "";
        const end = blockEnd(this.capture.buf);
        if (end >= 0) {
          const block = this.capture.buf.slice(0, end);
          const suffix = this.capture.buf.slice(end);
          this.capture = null;
          const calls = parseToolBlock(block, this.toolNames);
          if (calls.length) events.push({ type: "calls", calls });
          else events.push({ type: "text", text: block });
          buf = suffix;
          continue;
        }
        if (this.capture.buf.length > this.max) {
          events.push({ type: "text", text: this.capture.buf });
          this.capture = null;
          continue;
        }
        break; // 块未闭合，等更多数据
      }
      const idx = findStart(buf);
      if (idx < 0) {
        const hold = holdIndex(buf);
        if (hold >= 0) {
          if (hold > 0) events.push({ type: "text", text: buf.slice(0, hold) });
          this.pending = buf.slice(hold);
        } else {
          events.push({ type: "text", text: buf });
        }
        buf = "";
        break;
      }
      if (idx > 0) events.push({ type: "text", text: buf.slice(0, idx) });
      this.capture = { buf: buf.slice(idx) };
      buf = "";
    }
    return events;
  }

  // 流结束：未闭合的捕获区尝试宽松解析，失败当正文
  flush() {
    const events = [];
    if (this.capture) {
      const calls = parseToolBlock(this.capture.buf, this.toolNames, true);
      events.push(calls.length ? { type: "calls", calls } : { type: "text", text: this.capture.buf });
      this.capture = null;
    }
    if (this.pending) {
      events.push({ type: "text", text: this.pending });
      this.pending = "";
    }
    return events;
  }
}

// ---- prompt 组装 ----
//
// buildPromptParts 把拍平结果拆成 record（随对话增长的前缀）与 instructions（每轮重发的规则），
// 会话复用时只发 record 的增量尾部 + instructions（见 index.mjs 的会话缓存）。

// 超预算裁剪：从最老的轮次开始丢，保护最后一条用户/工具消息及其之后的所有轮次
function pruneTurns(turns, budgetTokens) {
  if (!budgetTokens || budgetTokens <= 0) return turns;
  const join = (list) => list.map((t) => `${t.role}：${t.text ?? ""}`).join("\n");
  let list = turns;
  let guard = list.length + 1;
  while (guard-- > 0) {
    const total = estimateTokens(join(list));
    if (total <= budgetTokens) break;
    let lastProt = -1;
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].kind === "user" || list[i].kind === "tool") {
        lastProt = i;
        break;
      }
    }
    if (lastProt <= 0) break; // 没有可丢的
    list = list.slice(1);
  }
  return list;
}

// 把 OpenAI chat completions 的 messages 拍平成网页版的 record + instructions 两段。
// 带 tools 时进入工具模式（DSML 协议），插件从回答流里筛出调用块转成 OpenAI tool_calls。
export function buildPromptParts(messages, tools, opts = {}) {
  const toolList = (Array.isArray(tools) ? tools : []).filter((t) => t?.type === "function" && t.function?.name);
  const hasTools = toolList.length > 0;

  // tool 消息常常只有 tool_call_id 没有 name，先建 id → 工具名 的映射
  const callNames = new Map();
  for (const m of messages ?? []) {
    if (m?.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const c of m.tool_calls) {
        if (c?.id && c.function?.name) callNames.set(c.id, c.function.name);
      }
    }
  }

  const toolLimit = Number(opts.toolResultLimit) > 0 ? Number(opts.toolResultLimit) : 0;
  const systems = [];
  const turns = [];
  for (const m of messages ?? []) {
    const text = contentToText(m.content);
    if (m.role === "system" || m.role === "developer") {
      if (text) systems.push(text);
    } else if (m.role === "assistant") {
      let t = text;
      if (Array.isArray(m.tool_calls) && m.tool_calls.length) {
        const rendered = hasTools
          ? formatToolCallsForPrompt(m.tool_calls)
          : `[助手${m.tool_calls.map((c) => `调用工具 ${c.function?.name ?? "?"}(${c.function?.arguments ?? ""})`).join("；")}]`;
        t = (t ? t + "\n" : "") + rendered;
      }
      turns.push({ role: "助手", text: t, kind: "assistant" });
    } else if (m.role === "tool") {
      const name = callNames.get(m.tool_call_id) ?? m.name ?? "工具";
      const body = toolLimit && text.length > toolLimit ? text.slice(0, toolLimit) + "\n…（截断）" : text;
      turns.push({ role: "用户", text: `[工具 ${name} 的返回结果]\n${body}`, kind: "tool" });
    } else {
      turns.push({ role: "用户", text, kind: "user" });
    }
  }
  const sysText = systems.join("\n\n");

  const renderRecord = (turnList) => {
    // 仅"无系统提示的单条用户消息"走直发；其余统一包装格式——
    // 保证同一对话每轮的 record 前缀一致，会话复用才能从第二轮开始发增量
    if (!sysText && !hasTools && turnList.length === 1 && turnList[0].role === "用户") {
      return turnList[0].text;
    }
    const lines = [];
    if (sysText) lines.push(`[系统设定]\n${sysText}\n`);
    lines.push("[对话记录]");
    for (const t of turnList) lines.push(`${t.role}：${t.text ?? ""}`);
    return lines.join("\n");
  };

  let kept = pruneTurns(turns, Number(opts.maxContextTokens) > 0 ? Number(opts.maxContextTokens) : 900000);
  const record = renderRecord(kept);
  const last = kept[kept.length - 1];

  let instructions;
  if (!hasTools) {
    if (kept.length === 1 && kept[0].role === "用户") {
      instructions = "";
    } else {
      const lastIsUser = last?.role === "用户";
      instructions = lastIsUser
        ? "请以助手的身份，直接回复上面对话记录中最后一条用户消息。不要复述记录本身。"
        : "请以助手的身份，直接接着上面最后一条助手消息继续说。不要复述记录本身。";
    }
  } else {
    const rules = [
      "[回复规则]",
      "你是上面对话记录里的助手本人，正在借助工具替用户完成任务。请从对话记录中断的地方继续：",
      last?.kind === "tool"
        ? "最后一条是工具返回结果。根据它决定下一步：要么调用下一个工具，要么给出给用户看的最终答复。"
        : last?.kind === "user"
          ? "请处理最后一条用户消息：需要动手就调用工具，能直接回答就直接回答。"
          : "接着最后的进度继续完成任务。",
    ];
    const toolInstr = buildToolInstructions(toolList, opts.toolChoice);
    instructions = `${toolInstr}\n\n${rules.join("\n")}`;
  }
  return { record, instructions };
}

export function buildPrompt(messages, tools, opts) {
  const { record, instructions } = buildPromptParts(messages, tools, opts);
  return instructions ? (record ? `${record}\n\n${instructions}` : instructions) : record;
}

// ---- SSE 解析 ----
//
// 网页版事件流（实测 2026-10）：
//   event: ready   data: {"response_message_id": 123, ...}        —— 多轮父链 id
//   event: hint    data: {"type":"error","finish_reason":"rate_limit_reached","content":"…频繁…"}
//   event: toast   data: {"type":"error", ...}                     —— 上游拒绝（如客户端过旧）
//   event: update_session|title|close
//   data: {"v": "增量文本"} / {"v": {"response": {...}}} / {"p":…,"o":"APPEND","v":…}
// THINK 分片 → reasoning，RESPONSE 分片 → content。
export class DeepSeekStreamParser {
  constructor() {
    this.event = null;
    this.dataLines = [];
    this.curType = null;
    this.finished = false;
    this.completionTokens = null;
    this.responseMessageId = null; // event: ready 帧里的多轮父链 id
    this.hintError = null; // { content, finishReason } —— hint/toast 的 type=error
  }

  feed(line) {
    const out = [];
    if (!line) {
      out.push(...this.flushEvent());
      return out;
    }
    if (line.startsWith(":")) return out;
    if (line.startsWith("event:")) {
      this.event = line.slice(6).trim();
      return out;
    }
    if (line.startsWith("data:")) {
      this.dataLines.push(line.slice(5).replace(/^ /, ""));
      return out;
    }
    return out;
  }

  flushEvent() {
    const out = [];
    let data = null;
    if (this.dataLines.length) {
      const raw = this.dataLines.join("\n");
      this.dataLines = [];
      try {
        data = JSON.parse(raw);
        out.push(...this.handleData(data));
      } catch {
        // 非 JSON 的 data 行（如 [DONE]）忽略
      }
    }
    if (this.event === "ready" && data && typeof data.response_message_id === "number") {
      this.responseMessageId = data.response_message_id;
    }
    if (
      (this.event === "hint" || this.event === "toast") &&
      data &&
      typeof data === "object" &&
      String(data.type ?? "").toLowerCase() === "error"
    ) {
      this.hintError = {
        event: this.event,
        content: String(data.content ?? data.msg ?? ""),
        finishReason: String(data.finish_reason ?? (this.event === "toast" ? "toast_error" : "hint_error")),
      };
    }
    if (this.event === "close") this.finished = true;
    this.event = null;
    return out;
  }

  emit(text, out) {
    if (!text) return;
    out.push(this.curType === "THINK" ? { type: "reasoning", text } : { type: "content", text });
  }

  handleData(obj) {
    const out = [];
    if (obj == null || typeof obj !== "object") return out;
    const { p, o, v } = obj;
    if (v === undefined) return out;
    if (p === undefined) {
      if (typeof v === "string") {
        this.emit(v, out);
      } else if (typeof v === "object") {
        const resp = v.response;
        if (resp) {
          for (const f of resp.fragments ?? []) {
            this.curType = f.type;
            this.emit(f.content, out);
          }
          if (typeof resp.accumulated_token_usage === "number") {
            this.completionTokens = resp.accumulated_token_usage;
          }
        }
      }
      return out;
    }
    if (o === "BATCH" && Array.isArray(v)) {
      for (const sub of v) {
        if (sub?.p === "accumulated_token_usage" && typeof sub.v === "number") {
          this.completionTokens = sub.v;
        }
      }
      return out;
    }
    if (p === "response/fragments" && o !== "SET") {
      for (const f of Array.isArray(v) ? v : [v]) {
        if (f && typeof f === "object") {
          this.curType = f.type;
          this.emit(f.content, out);
        }
      }
      return out;
    }
    if (typeof p === "string" && p.endsWith("/content") && typeof v === "string" && o !== "SET") {
      this.emit(v, out);
      return out;
    }
    if (p === "response/status" && v === "FINISHED") {
      this.finished = true;
    }
    return out;
  }
}

export class SseTranslator {
  constructor() {
    this.parser = new DeepSeekStreamParser();
    this.buf = "";
  }

  feed(text) {
    let buf = this.buf + text;
    const out = [];
    let idx;
    while ((idx = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, idx).replace(/\r$/, "");
      buf = buf.slice(idx + 1);
      out.push(...this.parser.feed(line));
      if (this.parser.finished) break;
    }
    this.buf = buf;
    return out;
  }

  get finished() {
    return this.parser.finished;
  }

  get completionTokens() {
    return this.parser.completionTokens;
  }

  get responseMessageId() {
    return this.parser.responseMessageId;
  }

  get hintError() {
    return this.parser.hintError;
  }
}
