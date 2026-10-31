// magpie-deepseek-web —— 把 DeepSeek 网页版（chat.deepseek.com）代理成 magpie 供应商。
//
// 网页版协议（已实测，2026-10）：
//   鉴权：Authorization: Bearer <userToken>（网页版 localStorage 里的 userToken）
//   流程：POST /api/v0/chat_session/create → POST /api/v0/chat/create_pow_challenge
//         → 本地解 PoW（DeepSeekHashV1，官方 sha3 wasm）→ POST /api/v0/chat/completion（SSE）
//   PoW 头：x-ds-pow-response = base64(JSON{algorithm,challenge,salt,answer,signature,target_path})
//   其余 x-client-* / x-hif-* 头服务端不强制要求。
//
// magpie 的网关把智能体的请求转成 OpenAI chat completions 发进这里的 fetch，
// 本插件负责转成网页版格式、再把它的事件流补丁转回 OpenAI chunk。

import { powResponseHeader } from "./pow.mjs";
import { buildPrompt, SseTranslator, ToolCallStream } from "./lib.mjs";

const PROVIDER = "deepseek-web";
const BASE = "https://chat.deepseek.com/api/v0";

const MODEL_DEFS = {
  "deepseek-chat": { name: "DeepSeek（网页版）", thinking: false, reasoning: false },
  "deepseek-reasoner": { name: "DeepSeek 深度思考（网页版）", thinking: true, reasoning: true },
};

function modelDef(model) {
  if (typeof model === "string" && model.includes("reasoner")) return MODEL_DEFS["deepseek-reasoner"];
  return MODEL_DEFS["deepseek-chat"];
}

async function postJson(token, path, body, signal) {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    signal,
    headers: { "content-type": "application/json", authorization: `Bearer ${token}` },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch {}
  return { status: r.status, text, json };
}

async function deleteSession(token, sessionId) {
  if (!sessionId) return;
  try {
    await postJson(token, "/chat_session/delete", { chat_session_ids: [sessionId] });
  } catch {}
}

function jsonError(status, message, signIn) {
  const headers = { "content-type": "application/json" };
  if (signIn) headers["x-magpie-sign-in"] = signIn;
  return new Response(JSON.stringify({ error: { message, type: "deepseek_web_error" } }), { status, headers });
}

// DeepSeek 有些失败用 HTTP 200 + JSON 业务码返回，也有些直接 401
function looksLikeAuthFailure(status, text) {
  if (status === 401) return true;
  return /\b40100\b|\b40101\b|\b40003\b|INVALID_TOKEN|unauthorized|token.{0,12}(invalid|expired|无效|过期)/i.test(text.slice(0, 500));
}

// 上游偶发 200 + 空事件流（无任何分片就 close）。当作成功会让网关只报
// "completed response with no content"，这里统一转成显式错误。
const EMPTY_RESPONSE_MSG = "DeepSeek 返回了空回答（可能触发风控、上下文超限或服务端瞬时异常），请重试或换模型";

// 把 completion 的 SSE 转成 OpenAI chat completions 的回答。
// enableTools 时从正文流里提取 <tool_call> 块转成 tool_calls（见 lib.mjs 的 ToolCallStream）。
function translateCompletion(upstream, { modelName, wantStream, includeUsage, token, sessionId, enableTools }) {
  const id = "chatcmpl-" + crypto.randomUUID().replaceAll("-", "").slice(0, 24);
  const created = Math.floor(Date.now() / 1000);
  const tr = new SseTranslator();
  const tc = enableTools ? new ToolCallStream() : null;
  const dec = new TextDecoder();
  const acc = wantStream ? null : { content: "", reasoning: "" };
  const emittedCalls = []; // { id, name, arguments }
  let callIndex = 0;
  let emittedAny = false; // 是否产出过任何正文/思考/调用，用于识别上游空流

  const chunkWith = (delta, finishReason = null, extra = {}) => ({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelName,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...extra,
  });

  function emitCall(call, sink) {
    const cid = "call_" + crypto.randomUUID().replaceAll("-", "").slice(0, 24);
    emittedCalls.push({ id: cid, name: call.name, arguments: call.arguments });
    emittedAny = true;
    if (!acc && sink) {
      sink(
        chunkWith({
          tool_calls: [
            { index: callIndex++, id: cid, type: "function", function: { name: call.name, arguments: call.arguments } },
          ],
        }),
      );
    }
  }

  // 处理一批增量：流式写入 sink，非流式累积
  function absorb(pieces, sink) {
    for (const piece of pieces) {
      if (piece.type === "reasoning") {
        if (acc) acc.reasoning += piece.text;
        else sink(chunkWith({ reasoning_content: piece.text }));
        if (piece.text) emittedAny = true;
        continue;
      }
      if (!tc) {
        if (acc) acc.content += piece.text;
        else sink(chunkWith({ content: piece.text }));
        if (piece.text) emittedAny = true;
        continue;
      }
      const { text, calls } = tc.feed(piece.text);
      if (acc) acc.content += text;
      else if (text) sink(chunkWith({ content: text }));
      if (text || calls.length) emittedAny = true;
      for (const c of calls) emitCall(c, sink);
    }
  }

  // 流收尾：提取残余（流式模式要在 finalize 之前先消化返回的正文/调用）
  function drainTail(sink) {
    if (!tc) return;
    const tail = tc.end();
    if (acc) acc.content += tail.text;
    else if (tail.text && sink) sink(chunkWith({ content: tail.text }));
    if (tail.text || tail.calls.length) emittedAny = true;
    for (const c of tail.calls) emitCall(c, sink);
  }

  function finalize() {
    drainTail(null);
    // 网页版只给会话累计用量（accumulated_token_usage），拆不出输入/输出，只放进 total_tokens
    const usage =
      tr.completionTokens != null
        ? { prompt_tokens: 0, completion_tokens: 0, total_tokens: tr.completionTokens }
        : undefined;
    if (!emittedAny) {
      // 上游 200 + 空事件流（实测偶发：风控/超限/服务端瞬时异常都会这样静默）。
      // 不能当成功返回，否则网关只会报 "no content"，看不见真正的问题。
      if (acc) return null;
      return [
        `data: ${JSON.stringify({ error: { message: EMPTY_RESPONSE_MSG, type: "deepseek_web_error" } })}\n\n`,
        "data: [DONE]\n\n",
      ].join("");
    }
    if (acc) {
      const message = { role: "assistant", content: acc.content || null };
      if (acc.reasoning) message.reasoning_content = acc.reasoning;
      if (emittedCalls.length) {
        message.tool_calls = emittedCalls.map((c) => ({
          id: c.id,
          type: "function",
          function: { name: c.name, arguments: c.arguments },
        }));
      }
      return {
        id,
        object: "chat.completion",
        created,
        model: modelName,
        choices: [{ index: 0, message, finish_reason: emittedCalls.length ? "tool_calls" : "stop" }],
        ...(usage ? { usage } : {}),
      };
    }
    const finish = emittedCalls.length ? "tool_calls" : "stop";
    const parts = [`data: ${JSON.stringify(chunkWith({}, finish))}\n\n`];
    if (includeUsage && usage) {
      parts.push(
        `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: modelName, choices: [], usage })}\n\n`,
      );
    }
    parts.push("data: [DONE]\n\n");
    return parts.join("");
  }

  // 喂入上游原始字节文本，返回要写进响应流的字符串
  function feed(text) {
    const out = [];
    absorb(tr.feed(text), (c) => out.push(`data: ${JSON.stringify(c)}\n\n`));
    return out.join("");
  }

  if (!wantStream) {
    return { mode: "buffer", feed, finalize };
  }

  const body = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      const push = (s) => controller.enqueue(enc.encode(s));
      const sink = (c) => push(`data: ${JSON.stringify(c)}\n\n`);
      const reader = upstream.body.getReader();
      try {
        push(`data: ${JSON.stringify(chunkWith({ role: "assistant", content: "" }))}\n\n`);
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const s = feed(dec.decode(value, { stream: true }));
          if (s) push(s);
          if (tr.finished) break;
        }
        drainTail(sink);
        push(finalize());
        controller.close();
      } catch (e) {
        controller.error(e);
      } finally {
        deleteSession(token, sessionId);
      }
    },
    cancel() {
      deleteSession(token, sessionId);
    },
  });
  return { mode: "stream", body };
}

async function handleChatCompletion(init, token, options) {
  let req;
  try {
    req = JSON.parse(typeof init?.body === "string" ? init.body : "{}");
  } catch {
    return jsonError(400, "请求体不是合法 JSON");
  }
  const modelName = req.model ?? "deepseek-chat";
  const def = modelDef(modelName);
  const wantStream = req.stream !== false;
  const includeUsage = req.stream_options?.include_usage === true;
  const messages = Array.isArray(req.messages) ? req.messages : [];
  const enableTools =
    req.tool_choice !== "none" &&
    Array.isArray(req.tools) &&
    req.tools.some((t) => t?.type === "function" && t.function?.name);
  const prompt = buildPrompt(messages, enableTools ? req.tools : null);
  const signal = init?.signal;

  const session = await postJson(token, "/chat_session/create", {}, signal);
  const sessionId = session.json?.data?.biz_data?.chat_session?.id;
  if (session.status !== 200 || !sessionId) {
    if (looksLikeAuthFailure(session.status, session.text)) {
      return jsonError(401, "DeepSeek 拒绝了这个 userToken，请重新登录", "expired");
    }
    return jsonError(502, `创建 DeepSeek 会话失败：HTTP ${session.status} ${session.text.slice(0, 200)}`);
  }

  try {
    const ch = await postJson(token, "/chat/create_pow_challenge", { target_path: "/api/v0/chat/completion" }, signal);
    const challenge = ch.json?.data?.biz_data?.challenge;
    if (ch.status !== 200 || !challenge) {
      if (looksLikeAuthFailure(ch.status, ch.text)) {
        return jsonError(401, "DeepSeek 拒绝了这个 userToken，请重新登录", "expired");
      }
      return jsonError(502, `获取 PoW 挑战失败：HTTP ${ch.status} ${ch.text.slice(0, 200)}`);
    }
    const header = await powResponseHeader(challenge);

    const upstream = await fetch(`${BASE}/chat/completion`, {
      method: "POST",
      signal,
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${token}`,
        "x-ds-pow-response": header,
      },
      body: JSON.stringify({
        chat_session_id: sessionId,
        parent_message_id: null,
        model_type: "default",
        prompt,
        ref_file_ids: [],
        thinking_enabled: def.thinking,
        search_enabled: options?.search === true,
        action: null,
        preempt: false,
      }),
    });

    if (!upstream.ok) {
      if (upstream.status === 401) {
        await deleteSession(token, sessionId);
        return jsonError(401, "DeepSeek 拒绝了这个 userToken，请重新登录", "expired");
      }
      // 其余非 2xx 原样透传，magpie 会按厂商回答处理限流与故障切换
      return new Response(upstream.body, { status: upstream.status, headers: upstream.headers });
    }

    const ct = upstream.headers.get("content-type") ?? "";
    if (!ct.includes("text/event-stream")) {
      const text = await upstream.text();
      await deleteSession(token, sessionId);
      if (looksLikeAuthFailure(200, text)) {
        return jsonError(401, "DeepSeek 拒绝了这个 userToken，请重新登录", "expired");
      }
      return jsonError(502, `DeepSeek 返回了异常回答：${text.slice(0, 400)}`);
    }

    const keep = options?.keepSessions === true;
    const t = translateCompletion(upstream, {
      modelName,
      wantStream,
      includeUsage,
      token,
      sessionId: keep ? null : sessionId,
      enableTools,
    });
    if (t.mode === "stream") {
      return new Response(t.body, {
        status: 200,
        headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" },
      });
    }
    // 非流式：读完后组装一个完整回答
    const dec = new TextDecoder();
    const reader = upstream.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      t.feed(dec.decode(value, { stream: true }));
    }
    const final = t.finalize();
    if (final === null) {
      return jsonError(502, EMPTY_RESPONSE_MSG);
    }
    return new Response(JSON.stringify(final), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  } catch (e) {
    if (options?.keepSessions !== true) await deleteSession(token, sessionId);
    throw e;
  }
}

export const DeepSeekWebPlugin = async (_input, options) => ({
  config: async (cfg) => {
    cfg.provider ??= {};
    cfg.provider[PROVIDER] ??= {
      name: "DeepSeek 网页版",
      npm: "@ai-sdk/openai-compatible", // 网关用 chat completions 格式发进来
      api: BASE,
      models: Object.fromEntries(
        Object.entries(MODEL_DEFS).map(([id, m]) => [
          id,
          {
            name: m.name,
            limit: { context: 131072, output: 8192 },
            reasoning: m.reasoning,
            tool_call: true, // 文本协议桥接：插件把工具定义写进 prompt 并解析 <tool_call> 块
            modalities: { input: ["text"] },
            cost: { input: 0, output: 0, cache_read: 0, cache_write: 0 },
            free: true,
          },
        ]),
      ),
    };
  },

  auth: {
    provider: PROVIDER,
    maxConcurrency: 3,
    methods: [
      {
        type: "api",
        label: "DeepSeek userToken",
        placeholder: "粘贴 chat.deepseek.com 网页版 localStorage 里的 userToken",
      },
    ],
    async authorize(inputs) {
      const key = (typeof inputs === "string" ? inputs : inputs?.key ?? inputs?.value ?? "").trim();
      if (!key) return { type: "failed", error: "请粘贴 userToken" };
      try {
        const r = await fetch(`${BASE}/users/current`, { headers: { authorization: `Bearer ${key}` } });
        if (r.status === 401) return { type: "failed", error: "userToken 无效或已过期" };
        if (!r.ok) return { type: "failed", error: `DeepSeek 返回 HTTP ${r.status}` };
        const j = await r.json();
        if (j?.code !== 0) return { type: "failed", error: `DeepSeek 校验失败：${j?.msg ?? j?.code ?? "未知错误"}` };
        const u = j.data?.biz_data ?? {};
        const name = u.email || u.mobile_number || u.id || "DeepSeek 账号";
        return { type: "success", key, metadata: { email: name, accountId: u.id, name: u.id_profile?.name } };
      } catch (e) {
        return { type: "failed", error: `无法连接 DeepSeek：${e?.message ?? e}` };
      }
    },

    async usage(getAuth) {
      const auth = await getAuth();
      if (auth?.type !== "api" || !auth.key) return { windows: [] };
      try {
        const r = await fetch(`${BASE}/users/current`, { headers: { authorization: `Bearer ${auth.key}` } });
        if (r.status === 401) return { error: "userToken 已失效，请重新登录", signIn: "expired", windows: [] };
        if (!r.ok) return { error: `DeepSeek 返回 HTTP ${r.status}`, windows: [] };
        const j = await r.json();
        const u = j?.data?.biz_data;
        return {
          plan: "网页版（免费额度）",
          user: u?.email || u?.mobile_number || "",
          windows: [],
        };
      } catch (e) {
        return { error: String(e?.message ?? e), windows: [] };
      }
    },

    async loader(getAuth) {
      const auth = await getAuth();
      if (auth?.type !== "api" || !auth.key) return {};
      return {
        baseURL: BASE,
        fetch: (input, init) => handleChatCompletion(init, auth.key, options),
      };
    },
  },
});
