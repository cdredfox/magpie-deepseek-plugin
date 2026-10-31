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
import { buildPrompt, SseTranslator, ToolCallStream, extractImages, extractAttachments } from "./lib.mjs";

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
    headers: { "content-type": "application/json", authorization: `Bearer ${token}`, ...BROWSER_HEADERS },
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

// ---- 文件上传（图片走 VISION 通道）----
// 实测协议（2026-10）：
//   create_pow_challenge(target_path=/api/v0/file/upload_file) → PoW
//   POST /file/upload_file（multipart，字段 file）→ { id, model_kind: "VISION", status: "PENDING" }
//   GET  /file/fetch_files?file_ids=…  轮询到非 PENDING（实测 ~2.5s SUCCESS）
//   completion：model_type="vision" + ref_file_ids=[原始 id]（无需 fork）
function authError(message) {
  return Object.assign(new Error(message), { auth: true });
}

async function getPowHeader(token, targetPath, signal) {
  const ch = await postJson(token, "/chat/create_pow_challenge", { target_path: targetPath }, signal);
  const challenge = ch.json?.data?.biz_data?.challenge;
  if (ch.status !== 200 || !challenge) {
    if (looksLikeAuthFailure(ch.status, ch.text)) throw authError("userToken 已失效，请重新登录");
    throw new Error(`获取 PoW 挑战失败：HTTP ${ch.status} ${ch.text.slice(0, 200)}`);
  }
  return powResponseHeader(challenge);
}

async function uploadFile(token, { mime, base64, url, name }, signal) {
  let bytes;
  let contentType = mime ?? "application/octet-stream";
  if (base64) {
    bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  } else {
    const r = await fetch(url, { signal });
    if (!r.ok) throw new Error(`下载图片失败：HTTP ${r.status} ${url.slice(0, 120)}`);
    bytes = new Uint8Array(await r.arrayBuffer());
    contentType = r.headers.get("content-type")?.split(";")[0] ?? contentType;
  }
  const pow = await getPowHeader(token, "/api/v0/file/upload_file", signal);
  const fd = new FormData();
  fd.append("file", new Blob([bytes], { type: contentType }), name);
  const r = await fetch(`${BASE}/file/upload_file`, {
    method: "POST",
    signal,
    headers: { authorization: `Bearer ${token}`, "x-ds-pow-response": pow, ...BROWSER_HEADERS },
    body: fd,
  });
  const text = await r.text();
  let j = null;
  try { j = JSON.parse(text); } catch {}
  const id = j?.data?.biz_data?.id;
  if (j?.code !== 0 || !id) {
    if (looksLikeAuthFailure(r.status, text)) throw authError("userToken 已失效，请重新登录");
    throw new Error(`上传文件失败：${text.slice(0, 200)}`);
  }
  return id;
}

async function waitFilesReady(token, ids, signal, timeoutMs = 20000) {
  const query = ids.map((i) => `file_ids=${encodeURIComponent(i)}`).join("&");
  for (let waited = 0; waited < timeoutMs; waited += 2500) {
    if (signal?.aborted) return;
    const r = await fetch(`${BASE}/file/fetch_files?${query}`, { headers: { authorization: `Bearer ${token}`, ...BROWSER_HEADERS }, signal }).catch(() => null);
    const j = await r?.json().catch(() => null);
    const files = j?.data?.biz_data?.files ?? [];
    const statuses = files.filter((f) => ids.includes(f.id)).map((f) => String(f.status ?? "").toUpperCase());
    if (statuses.length && statuses.every((s) => s && s !== "PENDING" && s !== "PARSING" && s !== "UPLOADING" && s !== "QUEUED")) return;
    await new Promise((res) => setTimeout(res, 2500));
  }
  // 超时不阻塞：参考实现 5 秒后即接受 PENDING 文件
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

// 风控画像：网页版浏览器都带这些头，裸 fetch（无 UA）在持续高强度调用下更容易被限流盯上
const BROWSER_HEADERS = {
  "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
  origin: "https://chat.deepseek.com",
  referer: "https://chat.deepseek.com/",
  "x-client-platform": "web",
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 单次尝试：把上游 SSE 增量转成 OpenAI chunk（流式输出 SSE 行数组，非流式累积成对象）。
// enableTools 时从正文流里提取 <tool_call> 块转成 tool_calls（见 lib.mjs 的 ToolCallStream）。
// 空流重试由外层驱动：每次尝试用全新会话，重试对客户端不可见（流式首块已先行发送）。
function makeAttempt({ id, created, modelName, wantStream, includeUsage, enableTools }) {
  const tr = new SseTranslator();
  const tc = enableTools ? new ToolCallStream() : null;
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

  function emitCall(call, out) {
    const cid = "call_" + crypto.randomUUID().replaceAll("-", "").slice(0, 24);
    emittedCalls.push({ id: cid, name: call.name, arguments: call.arguments });
    emittedAny = true;
    if (!acc) {
      out.push(
        `data: ${JSON.stringify(
          chunkWith({
            tool_calls: [
              { index: callIndex++, id: cid, type: "function", function: { name: call.name, arguments: call.arguments } },
            ],
          }),
        )}\n\n`,
      );
    }
  }

  // 处理一批增量：流式写入 out，非流式累积
  function absorb(pieces, out) {
    for (const piece of pieces) {
      if (piece.type === "reasoning") {
        if (acc) acc.reasoning += piece.text;
        else out.push(`data: ${JSON.stringify(chunkWith({ reasoning_content: piece.text }))}\n\n`);
        if (piece.text) emittedAny = true;
        continue;
      }
      if (!tc) {
        if (acc) acc.content += piece.text;
        else if (piece.text) out.push(`data: ${JSON.stringify(chunkWith({ content: piece.text }))}\n\n`);
        if (piece.text) emittedAny = true;
        continue;
      }
      const { text, calls } = tc.feed(piece.text);
      if (acc) acc.content += text;
      else if (text) out.push(`data: ${JSON.stringify(chunkWith({ content: text }))}\n\n`);
      if (text || calls.length) emittedAny = true;
      for (const c of calls) emitCall(c, out);
    }
  }

  // 流收尾：提取残余（消化返回的正文/调用）
  function drain() {
    const out = [];
    if (!tc) return out;
    const tail = tc.end();
    if (tail.text || tail.calls.length) emittedAny = true;
    if (acc) acc.content += tail.text;
    else if (tail.text) out.push(`data: ${JSON.stringify(chunkWith({ content: tail.text }))}\n\n`);
    for (const c of tail.calls) emitCall(c, out);
    return out;
  }

  function finalize() {
    const out = drain();
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
      ];
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
    out.push(`data: ${JSON.stringify(chunkWith({}, finish))}\n\n`);
    if (includeUsage && usage) {
      out.push(
        `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: modelName, choices: [], usage })}\n\n`,
      );
    }
    out.push("data: [DONE]\n\n");
    return out;
  }

  // 喂入上游原始字节文本，返回要写进响应流的 SSE 字符串数组（非流式返回 []）
  function feed(text) {
    const out = [];
    absorb(tr.feed(text), out);
    return out;
  }

  return {
    feed,
    finalize,
    get emittedAny() {
      return emittedAny;
    },
    get finished() {
      return tr.finished;
    },
  };
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

  // 附件：上传到网页版后用 ref_file_ids 引用；带图片时必须走 vision 模型后端。
  // 上传失败不静默降级——模型看不见图会胡编，显式报错让调用方重试。
  let refFileIds = [];
  let hasImage = false;
  try {
    const images = extractImages(messages);
    const attachments = extractAttachments(messages);
    hasImage = images.length > 0;
    for (const f of [...images, ...attachments]) {
      refFileIds.push(await uploadFile(token, f, signal));
    }
    if (refFileIds.length) await waitFilesReady(token, refFileIds, signal);
  } catch (e) {
    if (e?.auth) return jsonError(401, e.message, "expired");
    if (signal?.aborted) throw e;
    return jsonError(502, String(e?.message ?? e));
  }

  // 打开一次 completion 上游：全新会话 + PoW。失败以 err.auth（重新登录）/ err.passthrough
  // （非 2xx 原样透传给 magpie）区分，其余按普通错误处理。
  async function openUpstream() {
    const session = await postJson(token, "/chat_session/create", {}, signal);
    const sessionId = session.json?.data?.biz_data?.chat_session?.id;
    if (session.status !== 200 || !sessionId) {
      if (looksLikeAuthFailure(session.status, session.text)) throw authError("DeepSeek 拒绝了这个 userToken，请重新登录");
      throw new Error(`创建 DeepSeek 会话失败：HTTP ${session.status} ${session.text.slice(0, 200)}`);
    }
    const keep = options?.keepSessions === true;
    const cleanup = () => {
      if (keep !== true) deleteSession(token, sessionId);
    };
    try {
      const header = await getPowHeader(token, "/api/v0/chat/completion", signal);
      const upstream = await fetch(`${BASE}/chat/completion`, {
        method: "POST",
        signal,
        headers: { "content-type": "application/json", authorization: `Bearer ${token}`, "x-ds-pow-response": header, ...BROWSER_HEADERS },
        body: JSON.stringify({
          chat_session_id: sessionId,
          parent_message_id: null,
          model_type: hasImage ? "vision" : "default",
          prompt,
          ref_file_ids: refFileIds,
          thinking_enabled: def.thinking,
          search_enabled: options?.search === true,
          action: null,
          preempt: false,
        }),
      });
      if (!upstream.ok) {
        cleanup();
        if (upstream.status === 401) throw authError("DeepSeek 拒绝了这个 userToken，请重新登录");
        // 其余非 2xx 原样透传，magpie 会按厂商回答处理限流与故障切换
        throw Object.assign(new Error(`upstream ${upstream.status}`), {
          passthrough: new Response(upstream.body, { status: upstream.status, headers: upstream.headers }),
        });
      }
      const ct = upstream.headers.get("content-type") ?? "";
      if (!ct.includes("text/event-stream")) {
        const text = await upstream.text();
        cleanup();
        if (looksLikeAuthFailure(200, text)) throw authError("DeepSeek 拒绝了这个 userToken，请重新登录");
        throw new Error(`DeepSeek 返回了异常回答：${text.slice(0, 400)}`);
      }
      return { upstream, cleanup };
    } catch (e) {
      cleanup(); // 幂等：重复删除会话无害
      throw e;
    }
  }

  const dec = new TextDecoder();
  const retryAttempts = Math.max(1, Number(options?.retryAttempts) || 3);
  const retryBackoffMs = Math.max(0, Number(options?.retryBackoffMs) || 1500);
  // 把打开失败转成对外的 Response；流式重试中途失败无法再改状态码，走 error 事件
  const openOrFail = async () => {
    try {
      return await openUpstream();
    } catch (e) {
      if (e?.passthrough) return e.passthrough;
      if (e?.auth) return jsonError(401, e.message, "expired");
      if (signal?.aborted) throw e;
      return jsonError(502, String(e?.message ?? e));
    }
  };
  const isResponse = (x) => x instanceof Response;

  // 首次尝试先于响应承诺打开：错误能按原语义返回
  let opened = await openOrFail();
  if (isResponse(opened)) return opened;

  if (!wantStream) {
    // 非流式：整段读完；空流则换新会话退避重试
    for (let attempt = 1; ; attempt++) {
      const at = makeAttempt({ id: "chatcmpl-" + crypto.randomUUID().replaceAll("-", "").slice(0, 24), created: Math.floor(Date.now() / 1000), modelName, wantStream: false, includeUsage, enableTools });
      const reader = opened.upstream.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        at.feed(dec.decode(value, { stream: true }));
      }
      opened.cleanup();
      const final = at.finalize();
      if (final !== null || attempt >= retryAttempts || signal?.aborted) {
        if (final === null) return jsonError(502, EMPTY_RESPONSE_MSG);
        return new Response(JSON.stringify(final), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      await sleep(retryBackoffMs * attempt);
      opened = await openOrFail();
      if (isResponse(opened)) return opened;
    }
  }

  // 流式：空流时在流内换新会话重试——首块只有角色元数据，重试对客户端不可见
  const id = "chatcmpl-" + crypto.randomUUID().replaceAll("-", "").slice(0, 24);
  const created = Math.floor(Date.now() / 1000);
  let currentCleanup = opened.cleanup;
  const body = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      const push = (s) => controller.enqueue(enc.encode(s));
      try {
        push(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: modelName, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] })}\n\n`);
        for (let attempt = 1; ; attempt++) {
          const at = makeAttempt({ id, created, modelName, wantStream: true, includeUsage, enableTools });
          const reader = opened.upstream.body.getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            for (const s of at.feed(dec.decode(value, { stream: true }))) push(s);
            if (at.finished) break;
          }
          opened.cleanup();
          currentCleanup = null;
          if (!at.emittedAny && attempt < retryAttempts && !signal?.aborted) {
            await sleep(retryBackoffMs * attempt);
            opened = await openUpstream();
            currentCleanup = opened.cleanup;
            continue;
          }
          for (const s of at.finalize()) push(s);
          break;
        }
        controller.close();
      } catch (e) {
        try {
          push(`data: ${JSON.stringify({ error: { message: String(e?.message ?? e), type: "deepseek_web_error" } })}\n\n`);
          push("data: [DONE]\n\n");
          controller.close();
        } catch {
          controller.error(e);
        }
      }
    },
    cancel() {
      currentCleanup?.();
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache" },
  });
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
