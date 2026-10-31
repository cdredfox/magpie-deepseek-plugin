// magpie-deepseek-web —— 把 DeepSeek 网页版（chat.deepseek.com）代理成 magpie 的一个供应商。
//
// 网页版协议（已实测，2026-10）：
//   鉴权：Authorization: Bearer <userToken>（网页版 localStorage 里的 userToken）
//   流程：POST /api/v0/chat_session/create → POST /api/v0/chat/create_pow_challenge
//         → 本地解 PoW（DeepSeekHashV1，官方 sha3 wasm）→ POST /api/v0/chat/completion（SSE）
//   PoW 头：x-ds-pow-response = base64(JSON{algorithm,challenge,salt,answer,signature,target_path})
//   会话复用：event: ready 帧带 response_message_id，作为下一轮 parent_message_id，
//             配合增量 prompt（只发新增尾部）把 agent 循环的输入吞吐降一个数量级。
//   错误分类：SSE event: hint/toast (type=error) 与 JSON 200 的 biz_code=5（禁言）
//             都会伪装成"成功但空"，必须显式识别（参考 freeapi / web2api 两个逆向项目）。
//
// magpie 的网关把智能体的请求转成 OpenAI chat completions 发进这里的 fetch，
// 本插件负责转成网页版格式、再把它的事件流补丁转回 OpenAI chunk。

import { powResponseHeader } from "./pow.mjs";
import { buildPromptParts, SseTranslator, StreamSieve, extractImages, extractAttachments, estimateTokens, sanitizeLeaked } from "./lib.mjs";

const PROVIDER = "deepseek-web";
const BASE = "https://chat.deepseek.com/api/v0";

const MODEL_DEFS = {
  "deepseek-chat": { name: "DeepSeek（网页版）", thinking: false, search: false },
  "deepseek-reasoner": { name: "DeepSeek 深度思考（网页版）", thinking: true, search: false },
  "deepseek-search": { name: "DeepSeek 联网搜索（网页版）", thinking: false, search: true },
  "deepseek-reasoner-search": { name: "DeepSeek 深度思考+联网（网页版）", thinking: true, search: true },
};

function modelDef(model) {
  const id = String(model ?? "deepseek-chat");
  if (id.includes("reasoner-search")) return MODEL_DEFS["deepseek-reasoner-search"];
  if (id.includes("reasoner")) return MODEL_DEFS["deepseek-reasoner"];
  if (id.includes("search")) return MODEL_DEFS["deepseek-search"];
  return MODEL_DEFS["deepseek-chat"];
}

// 风控画像：补齐真实浏览器会带的头（参考 web2api 从 Chrome 抓包的头清单）
const BROWSER_HEADERS = {
  "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
  accept: "*/*",
  "accept-language": "zh-CN,zh;q=0.9,en;q=0.7",
  origin: "https://chat.deepseek.com",
  referer: "https://chat.deepseek.com/",
  priority: "u=1, i",
  "sec-ch-ua": '"Google Chrome";v="141", "Chromium";v="141", "Not_A Brand";v="24"',
  "sec-ch-ua-mobile": "?0",
  "sec-ch-ua-platform": '"macOS"',
  "sec-fetch-dest": "empty",
  "sec-fetch-mode": "cors",
  "sec-fetch-site": "same-origin",
  "x-client-version": "2.3.0",
  "x-client-platform": "web",
  "x-client-locale": "zh_CN",
  "x-client-timezone-offset": "28800",
  "x-client-bundle-id": "com.deepseek.chat",
};

const EMPTY_RESPONSE_MSG = "DeepSeek 返回了空回答（重试后仍为空；可能触发风控/限流冷却或服务端异常），请稍后重试或换模型";
const SESSION_TTL_MS = 30 * 60 * 1000; // 会话复用窗口：30 分钟
const MAX_CONTEXT_TOKENS = 900000; // 1M 留 10%（freeapi 的 session 续期阈值）

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function authError(message) {
  return Object.assign(new Error(message), { auth: true });
}

function muteError(message) {
  return Object.assign(new Error(message), { mute: true });
}

// DeepSeek 有些失败用 HTTP 200 + JSON 业务码返回，也有些直接 401
function looksLikeAuthFailure(status, text) {
  if (status === 401) return true;
  return /\b40100\b|\b40101\b|\b40003\b|INVALID_TOKEN|unauthorized|token.{0,12}(invalid|expired|无效|过期)/i.test(text.slice(0, 500));
}

// 账号禁言：普通 JSON 200（没有 SSE 通道），biz_code 5 / "user is muted"。
// 不识别的话它会伪装成"成功但空"（web2api 的 UserMutedError）。
function looksLikeMute(text) {
  return /"biz_code"\s*:\s*5\b|"user is muted"|"is_mute"\s*:\s*true|muted?_until/i.test(text.slice(0, 800));
}

// WAF/CDN 挑战：IP 层风控，换账号或稍后再试（插件层报显式错误）
function wafKind(response) {
  if (response.headers.get("x-amzn-waf-action")) return "aws-waf";
  if (response.headers.get("cf-mitigated") === "challenge") return "cloudflare";
  return null;
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

async function sha256Hex(bytes) {
  const h = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function getPowHeader(token, targetPath, signal, jitterMs) {
  if (jitterMs > 0) await sleep(Math.random() * jitterMs); // 打破匀速自动化节奏
  const ch = await postJson(token, "/chat/create_pow_challenge", { target_path: targetPath }, signal);
  const challenge = ch.json?.data?.biz_data?.challenge;
  if (ch.status !== 200 || !challenge) {
    if (looksLikeAuthFailure(ch.status, ch.text)) throw authError("userToken 已失效，请重新登录");
    throw new Error(`获取 PoW 挑战失败：HTTP ${ch.status} ${ch.text.slice(0, 200)}`);
  }
  return powResponseHeader(challenge);
}

async function deleteSession(token, sessionId) {
  if (!sessionId) return;
  try {
    await postJson(token, "/chat_session/delete", { chat_session_ids: [sessionId] });
  } catch {}
}

// ---- 文件上传（图片走 VISION 通道）----
//   create_pow_challenge(target_path=/api/v0/file/upload_file) → PoW
//   POST /file/upload_file（multipart，字段 file；X-File-Size/X-Model-Type/X-Thinking-Enabled）
//   GET /file/fetch_files?file_ids=…（重复参数，逗号分隔会被拒）轮询到非 PENDING
//   completion：ref_file_ids=[原始 id] + model_type="vision"
async function fetchFileBytes({ mime, base64, url }, signal, maxBytes) {
  if (base64) {
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    if (bytes.length > maxBytes) throw new Error(`图片超过大小限制：${bytes.length} > ${maxBytes} 字节`);
    return { bytes, contentType: mime ?? "application/octet-stream" };
  }
  const r = await fetch(url, { signal, headers: { accept: "image/*", "user-agent": BROWSER_HEADERS["user-agent"] } });
  if (!r.ok) throw new Error(`下载图片失败：HTTP ${r.status} ${url.slice(0, 120)}`);
  const bytes = new Uint8Array(await r.arrayBuffer());
  if (bytes.length > maxBytes) throw new Error(`图片超过大小限制：${bytes.length} > ${maxBytes} 字节`);
  return { bytes, contentType: r.headers.get("content-type")?.split(";")[0] ?? mime ?? "application/octet-stream" };
}

async function uploadBytes(token, bytes, contentType, name, signal, { modelType, thinking }) {
  const pow = await getPowHeader(token, "/api/v0/file/upload_file", signal, 0);
  const fd = new FormData();
  fd.append("file", new Blob([bytes], { type: contentType }), name);
  const r = await fetch(`${BASE}/file/upload_file`, {
    method: "POST",
    signal,
    headers: {
      authorization: `Bearer ${token}`,
      "x-ds-pow-response": pow,
      "x-file-size": String(bytes.length),
      "x-model-type": modelType,
      "x-thinking-enabled": thinking ? "1" : "0",
      ...BROWSER_HEADERS,
    },
    body: fd,
  });
  const text = await r.text();
  let j = null;
  try {
    j = JSON.parse(text);
  } catch {}
  const id = j?.data?.biz_data?.id;
  if (j?.code !== 0 || !id) {
    if (looksLikeAuthFailure(r.status, text)) throw authError("userToken 已失效，请重新登录");
    throw new Error(`上传文件失败：${text.slice(0, 200)}`);
  }
  return { id, digest: await sha256Hex(bytes) };
}

// 轮询解析状态；audit_result 拒绝 / error_code 单独分类
async function waitFilesReady(token, ids, signal, timeoutMs = 20000) {
  const query = ids.map((i) => `file_ids=${encodeURIComponent(i)}`).join("&");
  for (let waited = 0; waited < timeoutMs; waited += 2000) {
    if (signal?.aborted) return;
    const r = await fetch(`${BASE}/file/fetch_files?${query}`, { headers: { authorization: `Bearer ${token}`, ...BROWSER_HEADERS }, signal }).catch(() => null);
    const j = await r?.json().catch(() => null);
    const files = j?.data?.biz_data?.files ?? [];
    const mine = files.filter((f) => ids.includes(f.id));
    if (mine.length) {
      for (const f of mine) {
        const audit = String(f.audit_result ?? "").toLowerCase();
        if (["fail", "reject", "block"].includes(audit)) throw new Error("附件未通过 DeepSeek 内容审核");
        if (f.error_code) throw new Error(`附件处理失败（error_code=${f.error_code}）`);
      }
      const statuses = mine.map((f) => String(f.status ?? "").toUpperCase());
      if (statuses.length === ids.length && statuses.every((s) => s && s !== "PENDING" && s !== "PARSING" && s !== "UPLOADING" && s !== "QUEUED")) return;
    }
    await sleep(2000);
  }
  // 超时不阻塞：上游通常 2-3 秒完成，个别大文件稍后也能引用
}

// 图片/附件 digest 缓存：同账号下相同内容不重复上传（上游实测 file-id 可跨会话引用）
const fileCache = new Map(); // token → Map<digest, fileId>
const FILE_CACHE_MAX = 64;

function cacheFileId(token, digest, fileId) {
  let m = fileCache.get(token);
  if (!m) {
    m = new Map();
    fileCache.set(token, m);
  }
  m.set(digest, fileId);
  while (m.size > FILE_CACHE_MAX) {
    m.delete(m.keys().next().value);
  }
}

function jsonError(status, message, signIn) {
  const headers = { "content-type": "application/json" };
  if (signIn) headers["x-magpie-sign-in"] = signIn;
  return new Response(JSON.stringify({ error: { message, type: "deepseek_web_error" } }), { status, headers });
}

// ---- 会话复用缓存（增量提示）----
// 每账号最多保留 4 个会话（LRU）；新请求的 record 是缓存 record 的延伸 → 只发增量尾部；
// 完全相同 → 客户端重试，原样重发；发散（编辑历史/别的对话）→ 新建会话。
// 注意：复用意味着上游会话不再"用完即删"（TTL 30 分钟，可用 sessionReuse: false 关闭）。
const sessionCache = new Map(); // token → Entry[]（最多 4）
const SESSION_CACHE_MAX = 4;

function cacheEvictOldest(token, keepSessions) {
  const arr = sessionCache.get(token);
  if (!arr || arr.length < SESSION_CACHE_MAX) return;
  const old = arr.pop();
  if (!keepSessions) deleteSession(token, old.sessionId);
}

function cachePut(token, entry, keepSessions) {
  let arr = sessionCache.get(token) ?? [];
  const idx = arr.findIndex((e) => e.key === entry.key);
  if (idx >= 0) arr.splice(idx, 1);
  cacheEvictOldest(token, keepSessions);
  arr.unshift(entry);
  sessionCache.set(token, arr);
}

function cacheInvalidate(token, sessionId) {
  const arr = sessionCache.get(token);
  if (!arr) return;
  const rest = arr.filter((e) => e.sessionId !== sessionId);
  if (rest.length) sessionCache.set(token, rest);
  else sessionCache.delete(token);
}

// 单次尝试：把上游 SSE 增量转成 OpenAI chunk（流式输出 SSE 行数组，非流式累积成对象）。
// enableTools 时用 StreamSieve 从正文流里筛出调用块（DSML / <tool_call> 都认）。
// 空流/限流的分类与重试由外层驱动；重试对客户端不可见（流式首块仅角色元数据）。
function makeAttempt({ id, created, modelName, wantStream, includeUsage, enableTools, toolNames }) {
  const tr = new SseTranslator();
  const sieve = enableTools ? new StreamSieve({ toolNames }) : null;
  const acc = wantStream ? null : { content: "", reasoning: "" };
  const emittedCalls = [];
  let callIndex = 0;
  let emittedAny = false;

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

  function absorb(pieces, out) {
    for (const piece of pieces) {
      if (piece.type === "reasoning") {
        if (acc) acc.reasoning += piece.text;
        else if (piece.text) out.push(`data: ${JSON.stringify(chunkWith({ reasoning_content: piece.text }))}\n\n`);
        if (piece.text) emittedAny = true;
        continue;
      }
      if (!sieve) {
        const text = sanitizeLeaked(piece.text);
        if (acc) acc.content += text;
        else if (text) out.push(`data: ${JSON.stringify(chunkWith({ content: text }))}\n\n`);
        if (text) emittedAny = true;
        continue;
      }
      for (const ev of sieve.feed(piece.text)) {
        if (ev.type === "text") {
          const text = sanitizeLeaked(ev.text);
          if (acc) acc.content += text;
          else if (text) out.push(`data: ${JSON.stringify(chunkWith({ content: text }))}\n\n`);
          if (text) emittedAny = true;
        } else {
          for (const c of ev.calls) emitCall(c, out);
        }
      }
    }
  }

  function drain() {
    const out = [];
    if (!sieve) return out;
    for (const ev of sieve.flush()) {
      if (ev.type === "text") {
        const text = sanitizeLeaked(ev.text);
        if (acc) acc.content += text;
        else if (text) out.push(`data: ${JSON.stringify(chunkWith({ content: text }))}\n\n`);
        if (text) emittedAny = true;
      } else {
        for (const c of ev.calls) emitCall(c, out);
      }
    }
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
      // 空流：非流式返回 null 让外层重试/报错；流式由外层决定重试或发 error 事件
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
    get hintError() {
      return tr.hintError;
    },
    get responseMessageId() {
      return tr.responseMessageId;
    },
  };
}

function isRateLimitHint(hint) {
  if (!hint) return false;
  return hint.finishReason === "rate_limit_reached" || /频繁/.test(hint.content ?? "");
}

function hintMessage(hint) {
  const c = hint?.content ? `：${hint.content}` : "";
  return `DeepSeek 上游提示错误（${hint?.finishReason ?? "unknown"}）${c}`;
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
  const toolNames = enableTools ? req.tools.map((t) => t.function?.name).filter(Boolean) : [];
  const signal = init?.signal;
  const keepSessions = options?.keepSessions === true;
  const reuseEnabled = options?.sessionReuse !== false;
  const maxContextTokens = Number(options?.maxContextTokens) > 0 ? Number(options?.maxContextTokens) : MAX_CONTEXT_TOKENS;
  const retryAttempts = Math.max(1, Number(options?.retryAttempts) || 3);
  const retryBackoffMs = Math.max(0, Number(options?.retryBackoffMs) || 1500);
  const rateDelays = Array.isArray(options?.rateLimitRetryDelays)
    ? options.rateLimitRetryDelays.map((n) => Math.max(0, Number(n) || 0))
    : [5000, 15000];
  const jitterMs = Math.max(0, Number(options?.jitterMs) || 0);

  // 组装 prompt：record 随对话增长；instructions 每轮重发。
  // 超预算时**不在复用路径**存档（前缀已在上游会话里）——只有回落到新会话才拆存档上传。
  const buildOpts = {
    toolChoice: req.tool_choice,
    maxContextTokens,
    toolResultLimit: options?.toolResultLimit,
  };
  let parts = buildPromptParts(messages, enableTools ? req.tools : null, buildOpts);
  const assemble = (p) => (p.instructions ? (p.record ? `${p.record}\n\n${p.instructions}` : p.instructions) : p.record);
  let useArchive = false;
  let archiveFileId = null;
  const ensureArchive = async () => {
    if (!useArchive || archiveFileId) return;
    const bytes = new TextEncoder().encode(parts.archive);
    const digest = await sha256Hex(bytes);
    let fileId = fileCache.get(token)?.get(digest);
    if (!fileId) {
      const up = await uploadBytes(token, bytes, "text/markdown", "对话历史存档.md", signal, {
        modelType: "default",
        thinking: false,
      });
      fileId = up.id;
      cacheFileId(token, digest, fileId);
    }
    archiveFileId = fileId;
    refFileIds = [fileId, ...refFileIds.filter((id) => id !== fileId)];
  };
  const promptFor = (attempt) => {
    if (attempt === 1 && reuseKind === "delta" && reuseEntry) {
      const tail = parts.record.slice(reuseEntry.record.length).replace(/^\n+/, "");
      return parts.instructions ? `${tail}\n\n${parts.instructions}` : tail;
    }
    // 新会话：超预算则带历史存档附件
    if (useArchive) {
      return `[历史对话已作为附件提供（见附件列表第一项），请先阅读附件了解前情。]\n${assemble(parts)}`;
    }
    return assemble(parts);
  };

  // 会话复用判定：record 延伸 → 增量；相同 → 重试原样重发；发散/超限 → 新会话
  let reuseEntry = null;
  let reuseKind = null;
  if (reuseEnabled && signal?.aborted !== true) {
    const entries = (sessionCache.get(token) ?? []).filter((e) => Date.now() - e.sentAt < SESSION_TTL_MS);
    let best = null;
    for (const e of entries) {
      if (parts.record === e.record) best = best && best.record.length >= e.record.length ? best : e;
      else if (parts.record.startsWith(e.record) && (!best || e.record.length > best.record.length)) best = e;
    }
    if (best && estimateTokens(parts.record) <= maxContextTokens * 0.95) {
      reuseEntry = best;
      reuseKind = parts.record === best.record ? "same" : "delta";
    }
  }
  // 无会话可复用且历史超预算：拆存档（新会话首条消息带历史附件）
  if (!reuseEntry && estimateTokens(parts.record) > maxContextTokens) {
    parts = buildPromptParts(messages, enableTools ? req.tools : null, { ...buildOpts, forceArchive: true });
    useArchive = true;
  }

  // 附件：上传（digest 去重）→ 轮询解析 → ref_file_ids；带图片走 vision 后端。
  // 上传失败不静默降级——模型看不见图会胡编，显式报错让调用方重试。
  let refFileIds = [];
  let hasImage = false;
  try {
    const images = extractImages(messages);
    const attachments = extractAttachments(messages);
    hasImage = images.length > 0;
    const files = [...images, ...attachments];
    const maxFileBytes = 10 * 1024 * 1024;
    let totalBytes = 0;
    for (const f of files) {
      const { bytes, contentType } = await fetchFileBytes(f, signal, maxFileBytes);
      totalBytes += bytes.length;
      if (totalBytes > 32 * 1024 * 1024) throw new Error("附件总大小超过限制（32MiB）");
      const digest = await sha256Hex(bytes);
      let fileId = fileCache.get(token)?.get(digest);
      if (!fileId) {
        const up = await uploadBytes(token, bytes, contentType, f.name, signal, {
          modelType: hasImage ? "vision" : "default",
          thinking: def.thinking,
        });
        fileId = up.id;
        cacheFileId(token, digest, fileId);
      }
      refFileIds.push(fileId);
    }
    if (refFileIds.length) await waitFilesReady(token, refFileIds, signal);
  } catch (e) {
    if (e?.auth) return jsonError(401, e.message, "expired");
    if (signal?.aborted) throw e;
    return jsonError(502, String(e?.message ?? e));
  }

  // 打开一次 completion 上游。plan.reused 时复用缓存会话（不新建）。
  // 失败以 err.auth（重新登录）/ err.passthrough（非 2xx 透传）区分。
  const reuseParentId = reuseEntry?.parentMessageId ?? null;
  const attemptPrompt = (attempt) => {
    if (attempt === 1 && reuseKind === "delta" && reuseEntry) {
      const tail = parts.record.slice(reuseEntry.record.length).replace(/^\n+/, "");
      return parts.instructions ? `${tail}\n\n${parts.instructions}` : tail;
    }
    return promptFull; // fresh / retry-same / 重试一律完整 prompt（新会话没有父链历史）
  };
  async function openUpstream(attempt) {
    const reused = attempt === 1 && !!reuseEntry;
    let sessionId = reused ? reuseEntry.sessionId : null;
    if (!sessionId) {
      const session = await postJson(token, "/chat_session/create", {}, signal);
      sessionId = session.json?.data?.biz_data?.chat_session?.id;
      if (session.status !== 200 || !sessionId) {
        if (looksLikeAuthFailure(session.status, session.text)) throw authError("DeepSeek 拒绝了这个 userToken，请重新登录");
        throw new Error(`创建 DeepSeek 会话失败：HTTP ${session.status} ${session.text.slice(0, 200)}`);
      }
    }
    const discard = () => {
      if (reused) cacheInvalidate(token, sessionId);
      if (!keepSessions) deleteSession(token, sessionId);
    };
    try {
      if (!reused) await ensureArchive();
      const header = await getPowHeader(token, "/api/v0/chat/completion", signal, jitterMs);
      const body = {
        chat_session_id: sessionId,
        parent_message_id: reused ? reuseParentId : null,
        model_type: hasImage ? "vision" : "default",
        prompt: promptFor(attempt),
        ref_file_ids: refFileIds,
        thinking_enabled: def.thinking,
        search_enabled: def.search === true || options?.search === true,
        action: null,
        preempt: false,
      };
      if (typeof req.temperature === "number") body.temperature = req.temperature;
      if (typeof req.top_p === "number") body.top_p = req.top_p;
      if (typeof req.max_tokens === "number") body.max_tokens = req.max_tokens;
      const upstream = await fetch(`${BASE}/chat/completion`, {
        method: "POST",
        signal,
        headers: { "content-type": "application/json", authorization: `Bearer ${token}`, "x-ds-pow-response": header, ...BROWSER_HEADERS },
        body: JSON.stringify(body),
      });
      if (!upstream.ok) {
        discard();
        const waf = wafKind(upstream);
        if (waf) throw new Error(`DeepSeek WAF 挑战（${waf}），该出口 IP 被风控，请稍后重试或更换网络`);
        if (upstream.status === 401) throw authError("DeepSeek 拒绝了这个 userToken，请重新登录");
        // 其余非 2xx 原样透传，magpie 会按厂商回答处理限流与故障切换
        throw Object.assign(new Error(`upstream ${upstream.status}`), {
          passthrough: new Response(upstream.body, { status: upstream.status, headers: upstream.headers }),
        });
      }
      const ct = upstream.headers.get("content-type") ?? "";
      if (!ct.includes("text/event-stream")) {
        const text = await upstream.text();
        discard();
        if (looksLikeAuthFailure(200, text)) throw authError("DeepSeek 拒绝了这个 userToken，请重新登录");
        if (looksLikeMute(text)) {
          const until = /"muted?_until"\s*:\s*(\d+)/.exec(text)?.[1];
          throw muteError(`DeepSeek 账号被禁言${until ? `（至 ${new Date(Number(until) * 1000).toLocaleString()}）` : ""}，请换账号或稍后再试`);
        }
        throw new Error(`DeepSeek 返回了异常回答：${text.slice(0, 400)}`);
      }
      return {
        upstream,
        finish: (ok, responseMessageId) => {
          if (ok && reuseEnabled) {
            cachePut(
              token,
              {
                key: `${parts.virtualRecord.length}:${parts.virtualRecord.slice(0, 256)}`,
                record: parts.virtualRecord,
                sessionId,
                parentMessageId: responseMessageId ?? null,
                sentAt: Date.now(),
              },
              keepSessions,
            );
          } else if (!ok) {
            discard();
          } else if (!keepSessions) {
            deleteSession(token, sessionId); // 不复用：保持"用完即删"
          }
        },
      };
    } catch (e) {
      if (!(e?.passthrough || e?.auth || e?.mute)) discard();
      throw e;
    }
  }

  const dec = new TextDecoder();
  const isResponse = (x) => x instanceof Response;
  // 打开失败转成对外 Response；流式重试中途失败无法改状态码，走 error 事件
  async function openOrFail(attempt) {
    try {
      return await openUpstream(attempt);
    } catch (e) {
      if (e?.passthrough) return e.passthrough;
      if (e?.auth) return jsonError(401, e.message, "expired");
      if (e?.mute) return jsonError(429, e.message);
      if (signal?.aborted) throw e;
      return jsonError(502, String(e?.message ?? e));
    }
  }
  // 分类一次尝试的结果：返回 { kind: "ok"|"rateLimit"|"empty", hint }
  function classify(at) {
    const hint = at.hintError;
    if (hint && isRateLimitHint(hint)) return { kind: "rateLimit", hint };
    if (!at.emittedAny) return { kind: "empty", hint };
    return { kind: "ok", hint };
  }

  // 首次尝试先于响应承诺打开：错误能按原语义返回
  let opened = await openOrFail(1);
  if (isResponse(opened)) return opened;
  let currentDiscard = () => opened.finish(false);

  if (!wantStream) {
    // 非流式：整段读完；限流退避重试、空流换会话重试
    let attempt = 1;
    let rateIdx = 0;
    let emptyCount = 0;
    for (;;) {
      const at = makeAttempt({ id: "chatcmpl-" + crypto.randomUUID().replaceAll("-", "").slice(0, 24), created: Math.floor(Date.now() / 1000), modelName, wantStream: false, includeUsage, enableTools, toolNames });
      const reader = opened.upstream.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        at.feed(dec.decode(value, { stream: true }));
      }
      const verdict = classify(at);
      if (verdict.kind === "ok") {
        opened.finish(true, at.responseMessageId);
        const final = at.finalize();
        return new Response(JSON.stringify(final), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      opened.finish(false);
      const hint = verdict.hint;
      if (verdict.kind === "rateLimit" && rateIdx < rateDelays.length) {
        await sleep(rateDelays[rateIdx++]);
        attempt++;
      } else if (verdict.kind === "empty" && emptyCount < Math.max(0, retryAttempts - 1) && !hint) {
        emptyCount++;
        await sleep(retryBackoffMs * emptyCount);
        attempt++;
      } else if (hint && verdict.kind === "empty") {
        // 空 + hint（非限流）：上游明确拒绝，重试无意义
        return jsonError(502, hintMessage(hint));
      } else {
        return jsonError(502, verdict.kind === "rateLimit" ? "DeepSeek 上游限流（消息发送过于频繁），已自动重试仍被拒，请稍后再试或增加账号" : EMPTY_RESPONSE_MSG);
      }
      if (estimateTokens(parts.record) > maxContextTokens) {
        // 回落新会话且历史超预算：把最老轮次打包成附件，prompt 只留近期部分
        parts = buildPromptParts(messages, enableTools ? req.tools : null, { ...buildOpts, forceArchive: true });
        useArchive = true;
        await ensureArchive();
      }
      opened = await openOrFail(attempt);
      if (isResponse(opened)) return opened;
    }
  }

  // 流式：空流/限流在流内换新会话重试——首块只有角色元数据，重试对客户端不可见。
  // finished 后继续把上游读到尾（排干），避免污染 keep-alive 连接产生后续空响应。
  const id = "chatcmpl-" + crypto.randomUUID().replaceAll("-", "").slice(0, 24);
  const created = Math.floor(Date.now() / 1000);
  const body = new ReadableStream({
    async start(controller) {
      const enc = new TextEncoder();
      const push = (s) => controller.enqueue(enc.encode(s));
      try {
        push(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created, model: modelName, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] })}\n\n`);
        let attempt = 1;
        let rateIdx = 0;
        let emptyCount = 0;
        for (;;) {
          const at = makeAttempt({ id, created, modelName, wantStream: true, includeUsage, enableTools, toolNames });
          const reader = opened.upstream.body.getReader();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (!at.finished) {
              for (const s of at.feed(dec.decode(value, { stream: true }))) push(s);
            }
            // finished 后只排干不再转写
          }
          const verdict = classify(at);
          if (verdict.kind === "ok") {
            opened.finish(true, at.responseMessageId);
            for (const s of at.finalize()) push(s);
            break;
          }
          opened.finish(false);
          const hint = verdict.hint;
          const canRetry =
            !signal?.aborted &&
            (verdict.kind === "rateLimit"
              ? rateIdx < rateDelays.length
              : verdict.kind === "empty"
                ? emptyCount < Math.max(0, retryAttempts - 1) && !hint
                : false);
          if (canRetry) {
            if (verdict.kind === "rateLimit") await sleep(rateDelays[rateIdx++]);
            else {
              emptyCount++;
              await sleep(retryBackoffMs * emptyCount);
            }
            attempt++;
            if (estimateTokens(parts.record) > maxContextTokens) {
              parts = buildPromptParts(messages, enableTools ? req.tools : null, { ...buildOpts, forceArchive: true });
              useArchive = true;
            }
            await ensureArchive();
            opened = await openUpstream(attempt);
            currentDiscard = () => opened.finish(false);
            continue;
          }
          // 终态：发明确的 error 事件（SDK 能看到真实原因）
          const message =
            verdict.kind === "rateLimit"
              ? "DeepSeek 上游限流（消息发送过于频繁），已自动重试仍被拒，请稍后再试或增加账号"
              : hint
                ? hintMessage(hint)
                : EMPTY_RESPONSE_MSG;
          push(`data: ${JSON.stringify({ error: { message, type: "deepseek_web_error" } })}\n\n`);
          push("data: [DONE]\n\n");
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
      currentDiscard?.();
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
            // 网页版实测输入上限 2621440 字符 ≈ 1M token（client/settings?scope=model）
            limit: { context: 1000000, output: 65536 },
            reasoning: m.thinking,
            tool_call: true, // DSML 文本协议桥接：插件把工具定义写进 prompt 并筛分调用块
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
    maxConcurrency: 2, // 并发过高会触发账号禁言（web2api 同款默认）
    methods: [
      {
        type: "api",
        label: "DeepSeek userToken",
        placeholder: "粘贴 chat.deepseek.com 网页版 localStorage 里的 userToken",
      },
    ],
    async authorize(inputs) {
      // 容错粘贴：裸 token / "Bearer " 前缀 / localStorage 的 {"value":…} 包裹 / 引号
      let key = (typeof inputs === "string" ? inputs : inputs?.key ?? inputs?.value ?? "").trim();
      if (key.startsWith("Bearer ")) key = key.slice(7).trim();
      try {
        const j = JSON.parse(key);
        if (j && typeof j.value === "string") key = j.value.trim();
      } catch {}
      key = key.replace(/^"+|"+$/g, "").trim();
      if (!key) return { type: "failed", error: "请粘贴 userToken" };
      try {
        const r = await fetch(`${BASE}/users/current`, { headers: { authorization: `Bearer ${key}`, ...BROWSER_HEADERS } });
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
        const r = await fetch(`${BASE}/users/current`, { headers: { authorization: `Bearer ${auth.key}`, ...BROWSER_HEADERS } });
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
