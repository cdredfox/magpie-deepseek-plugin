// 纯函数部分：messages 拍平成网页版 prompt、DeepSeek SSE → OpenAI chunk 转换。

function contentToText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts = [];
  for (const p of content) {
    if (p?.type === "text" && p.text) parts.push(p.text);
    else if (p?.type === "image" || p?.type === "image_url") parts.push("[图片（网页版暂不支持）]");
    else if (typeof p?.text === "string") parts.push(p.text);
  }
  return parts.join("");
}

// 把一条历史里的工具调用渲染成协议块（和模型被要求输出的格式完全一致，方便模仿）
function callToProtocol(c) {
  const name = c.function?.name ?? "?";
  let raw = c.function?.arguments ?? "{}";
  if (typeof raw === "string") {
    try {
      raw = JSON.stringify(JSON.parse(raw));
    } catch {}
  } else {
    try {
      raw = JSON.stringify(raw ?? {});
    } catch {
      raw = "{}";
    }
  }
  return `<tool_call>\n{"name": ${JSON.stringify(name)}, "arguments": ${raw}}\n</tool_call>`;
}

// 把 OpenAI chat completions 的 messages 拍平成 DeepSeek 网页版的单条 prompt。
// 网页版接口没有 system 槽位、也吃不到完整历史（服务端会话不归我们管），
// 多轮对话按转写文本的方式塞进同一条 prompt。
// 带 tools 时进入工具模式：附上工具定义与调用协议（<tool_call> 块），
// 插件随后从回答流里解析这些块转成 OpenAI tool_calls。
export function buildPrompt(messages, tools) {
  const hasTools = Array.isArray(tools) && tools.some((t) => t?.type === "function" && t.function?.name);

  // tool 消息常常只有 tool_call_id 没有 name，先建 id → 工具名 的映射
  const callNames = new Map();
  for (const m of messages ?? []) {
    if (m?.role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const c of m.tool_calls) {
        if (c?.id && c.function?.name) callNames.set(c.id, c.function.name);
      }
    }
  }

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
          ? m.tool_calls.map(callToProtocol).join("\n")
          : `[助手${m.tool_calls.map((c) => `调用工具 ${c.function?.name ?? "?"}(${c.function?.arguments ?? ""})`).join("；")}]`;
        t = (t ? t + "\n" : "") + rendered;
      }
      turns.push({ role: "助手", text: t, kind: "assistant" });
    } else if (m.role === "tool") {
      const name = callNames.get(m.tool_call_id) ?? m.name ?? "工具";
      turns.push({ role: "用户", text: `[工具 ${name} 的返回结果]\n${text}`, kind: "tool" });
    } else {
      turns.push({ role: "用户", text, kind: "user" });
    }
  }
  const sysText = systems.join("\n\n");

  if (!hasTools) {
    if (turns.length === 1 && turns[0].role === "用户") {
      return sysText ? `${sysText}\n\n${turns[0].text}` : turns[0].text;
    }
    const lastIsUser = turns[turns.length - 1]?.role === "用户";
    const lines = [];
    if (sysText) lines.push(`[系统设定]\n${sysText}\n`);
    lines.push("[对话记录]");
    for (const t of turns) lines.push(`${t.role}：${t.text ?? ""}`);
    lines.push("");
    lines.push(
      lastIsUser
        ? "请以助手的身份，直接回复上面对话记录中最后一条用户消息。不要复述记录本身。"
        : "请以助手的身份，直接接着上面最后一条助手消息继续说。不要复述记录本身。",
    );
    return lines.join("\n");
  }

  // 工具模式：记录 + 工具定义 + 回复规则（规则放最后，指令遵循最强）
  const toolDefs = tools
    .filter((t) => t?.type === "function" && t.function?.name)
    .map((t) => {
      const f = t.function;
      const schema = f.parameters ? JSON.stringify(f.parameters) : "";
      return `### ${f.name}\n${(f.description ?? "").trim()}${schema ? `\n参数 JSON Schema：${schema}` : ""}`;
    })
    .join("\n\n");

  const last = turns[turns.length - 1];
  const lines = [];
  if (sysText) lines.push(`[系统设定]\n${sysText}\n`);
  lines.push("[对话记录]");
  for (const t of turns) lines.push(`${t.role}：${t.text ?? ""}`);
  lines.push("");
  lines.push("[可用工具]");
  lines.push(toolDefs);
  lines.push("");
  lines.push("[回复规则]");
  lines.push("你是上面对话记录里的助手本人，正在借助工具替用户完成任务。请从对话记录中断的地方继续：");
  lines.push(
    last?.kind === "tool"
      ? "最后一条是工具返回结果。根据它决定下一步：要么调用下一个工具，要么给出给用户看的最终答复。"
      : last?.kind === "user"
        ? "请处理最后一条用户消息：需要动手就调用工具，能直接回答就直接回答。"
        : "接着最后的进度继续完成任务。",
  );
  lines.push("需要调用工具时，原样输出下面的调用块（不要放进代码块，前后不要附加解释文字）：");
  lines.push('<tool_call>\n{"name": "工具名", "arguments": {…符合该工具定义的参数…}}\n</tool_call>');
  lines.push("可以一次连续输出多个调用块；调用块输出完必须立即停笔。");
  lines.push(
    "严禁编造工具的返回结果：不要替“用户”写任何消息，不要自己写出“[工具…的返回结果]”，不要假装已经看到结果。真实结果由系统在下一轮附进对话记录。",
  );
  lines.push("不需要调用工具时，直接写给用户看的答复正文，不要输出调用块。不要复述对话记录。");
  return lines.join("\n");
}

// 把上游字节流按行喂给 DeepSeekStreamParser，容许任意分包。
export class SseTranslator {
  constructor() {
    this.parser = new DeepSeekStreamParser();
    this.buf = "";
  }

  // 喂入任意一段原始文本，返回 [{ type: "content"|"reasoning", text }]
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
}
//   event: ready|update_session|title|close
//   data: {"v": "增量文本"}                                  —— 追加到当前分片
//   data: {"v": {"response": {...fragments: [...]}}}          —— 消息头，含初始分片
//   data: {"p": "response/fragments", "o": "APPEND", "v": [...]} —— 新分片（THINK / RESPONSE）
//   data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "..."} —— 追加
//   data: {"p": "response", "o": "BATCH", "v": [...]}         —— 批量，含 token 用量
// THINK 分片 → reasoning_content，RESPONSE 分片 → content。
export class DeepSeekStreamParser {
  constructor() {
    this.event = null;
    this.dataLines = [];
    this.curType = null;
    this.finished = false;
    this.completionTokens = null;
  }

  // 喂入一行原始文本（不含换行），返回 [{ type: "content"|"reasoning", text }]
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
    // 其他行原样忽略
    return out;
  }

  flushEvent() {
    const out = [];
    if (this.dataLines.length) {
      const raw = this.dataLines.join("\n");
      this.dataLines = [];
      try {
        out.push(...this.handleData(JSON.parse(raw)));
      } catch {
        // 非 JSON 的 data 行（如 [DONE]）忽略
      }
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

// ---- 工具调用提取 ----
//
// 网页版模型只能吐文本，插件约定它用 <tool_call>…</tool_call> 块表达工具调用
// （见 buildPrompt 的回复规则），这个类从内容流里实时把块剥出来：
//   - 其余文本原样通过（正文增量）
//   - 每个块解析成 { name, arguments }（arguments 归一化为 JSON 字符串，OpenAI 格式）
//   - 容忍任意分包：未完整的 "<tool_call>" 前缀会扣住不发
//   - 容忍模型把块包进 ``` 围栏：紧贴块前的纯围栏行会被剥掉
//   - JSON 解析失败时把原文当正文放行，不吞内容
const CALL_OPEN = "<tool_call>";
const CALL_CLOSE = "</tool_call>";

function callHoldbackLen(s) {
  const max = Math.min(s.length, CALL_OPEN.length - 1);
  for (let n = max; n > 0; n--) {
    if (CALL_OPEN.startsWith(s.slice(s.length - n))) return n;
  }
  return 0;
}

// 判断一行（不含换行）是不是纯围栏行。"```xm" 这种没写完的也算（等下一批定性）。
function isFenceLine(line) {
  return /^[ \t]*`{1,3}[a-zA-Z0-9_-]*[ \t]*$/.test(line);
}

// 剥掉紧贴调用块的围栏行：从尾部往回删完整的围栏行（含换行），
// 最后再剥掉紧贴 OPEN 的围栏片段（如 "```xml<tool_call>" 连写在同一行）。
function stripTrailingFenceLines(s) {
  let end = s.length;
  for (;;) {
    if (end === 0 || s[end - 1] !== "\n") break;
    const nl = s.lastIndexOf("\n", end - 2);
    if (!isFenceLine(s.slice(nl + 1, end - 1))) break;
    end = nl + 1;
  }
  if (end > 0 && s[end - 1] !== "\n") {
    const nl = s.lastIndexOf("\n", end - 1);
    if (isFenceLine(s.slice(nl + 1, end))) end = nl + 1;
  }
  return s.slice(0, end);
}

export class ToolCallStream {
  constructor() {
    this.buf = "";
    this.inside = false;
    this.ended = false;
    this.afterClose = false; // 刚消费完一个调用块，先消化后面的空白行/包栏围栏行
  }

  feed(text) {
    this.buf += text;
    const out = { text: "", calls: [] };
    for (;;) {
      if (this.inside) {
        const end = this.buf.indexOf(CALL_CLOSE);
        if (end === -1) break; // 块没闭合，等更多数据
        this.addCall(this.buf.slice(0, end), out, true);
        this.buf = this.buf.slice(end + CALL_CLOSE.length);
        this.inside = false;
        this.afterClose = true;
        continue;
      }
      if (this.afterClose) {
        // 逐行消化调用块后面的空白行和包调用的裸/xml/json 围栏行，
        // 碰到第一行正经内容就交回正常流程。行没收尾前先扣住（等定性）。
        // 必须在找下一个 OPEN 之前做，否则"围栏 + 紧跟下一个调用"时围栏会被当成正文放行。
        let waiting = false;
        for (;;) {
          const nl = this.buf.indexOf("\n");
          const complete = nl !== -1;
          const line = complete ? this.buf.slice(0, nl) : this.buf;
          if (/^[ \t]*$/.test(line)) {
            if (!complete) {
              waiting = true;
              break;
            }
            this.buf = this.buf.slice(nl + 1); // 空白行：跳过
            continue;
          }
          if (isFenceLine(line)) {
            if (!complete) {
              waiting = true;
              break;
            }
            if (/^[ \t]*```(?:xml|json)?[ \t]*$/.test(line)) {
              this.buf = this.buf.slice(nl + 1); // 包调用的围栏行：丢弃
              continue;
            }
            this.afterClose = false; // 带语言标注的围栏是正经代码块，保留
            break;
          }
          this.afterClose = false; // 正经内容
          break;
        }
        if (waiting) break;
      }
      const idx = this.buf.indexOf(CALL_OPEN);
      if (idx !== -1) {
        out.text += stripTrailingFenceLines(this.buf.slice(0, idx));
        this.buf = this.buf.slice(idx + CALL_OPEN.length);
        this.inside = true;
        this.afterClose = false;
        continue;
      }
      // 没有完整 OPEN：只放行能唯一判定的部分，其余扣住——
      //   a) 尾部 "<tool_call>" 的前缀（连同紧贴的围栏行）
      //   b) 尾部连续的纯围栏行（完整与否都要等下一行才能定性）
      let p = this.buf.length - callHoldbackLen(this.buf);
      if (p < this.buf.length) {
        for (;;) {
          if (p === 0) break;
          if (this.buf[p - 1] === "\n") {
            const nl = this.buf.lastIndexOf("\n", p - 2);
            if (!isFenceLine(this.buf.slice(nl + 1, p - 1))) break;
            p = nl + 1;
            continue;
          }
          const nl = this.buf.lastIndexOf("\n", p - 1);
          if (!isFenceLine(this.buf.slice(nl + 1, p))) break;
          p = nl + 1;
          continue;
        }
      } else if (this.buf.length) {
        let end = this.buf.length;
        for (;;) {
          if (end === 0) break;
          if (this.buf[end - 1] === "\n") {
            const nl = this.buf.lastIndexOf("\n", end - 2);
            if (!isFenceLine(this.buf.slice(nl + 1, end - 1))) break;
            end = nl + 1;
            continue;
          }
          const nl = this.buf.lastIndexOf("\n", end - 1);
          if (!isFenceLine(this.buf.slice(nl + 1, end))) break;
          end = nl + 1;
          continue;
        }
        p = Math.min(p, end);
      }
      if (p > 0) {
        out.text += this.buf.slice(0, p);
        this.buf = this.buf.slice(p);
      }
      break;
    }
    return out;
  }

  // 流结束：把残余处理掉（幂等）
  end() {
    if (this.ended) return { text: "", calls: [] };
    this.ended = true;
    const out = { text: "", calls: [] };
    if (this.inside) {
      // 没等到闭合标签：试着按 JSON 抢救，失败则原文放行
      this.addCall(this.buf, out, false);
      this.buf = "";
      this.inside = false;
    } else {
      out.text += this.buf;
      this.buf = "";
    }
    return out;
  }

  // withClose：解析失败回放原文时是否补回闭合标签（流中途的块有，抢救残块没有）
  addCall(raw, out, withClose) {
    let obj = null;
    try {
      obj = JSON.parse(raw.trim());
    } catch {}
    if (obj && typeof obj === "object" && typeof obj.name === "string" && obj.name) {
      let args = obj.arguments ?? {};
      if (typeof args !== "string") {
        try {
          args = JSON.stringify(args);
        } catch {
          args = "{}";
        }
      }
      out.calls.push({ name: obj.name, arguments: args });
    } else {
      out.text += CALL_OPEN + raw + (withClose ? CALL_CLOSE : "");
    }
  }
}
