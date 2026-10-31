// DeepSeekHashV1 PoW 求解器：直接运行 DeepSeek 网页版自带的 sha3 wasm（sha3_wasm_bg.wasm）。
// 调用 ABI（wasm-bindgen）：
//   wasm_solve(retptr, challenge_ptr, challenge_len, prefix_ptr, prefix_len, difficulty: f64)
// retptr 指向 16 字节栈槽：+0 为 i32 状态（1 成功 / 0 失败），+8 为 f64 答案（nonce）。
// prefix 固定为 `${salt}_${expire_at}_`，nonce 由 wasm 从 0 起逐个尝试。

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

let exportsPromise = null;

function wasmExports() {
  if (!exportsPromise) {
    exportsPromise = (async () => {
      const file = path.join(path.dirname(fileURLToPath(import.meta.url)), "sha3_wasm_bg.wasm");
      const { instance } = await WebAssembly.instantiate(await readFile(file));
      return instance.exports;
    })();
  }
  return exportsPromise;
}

export async function solvePow({ challenge, salt, expire_at, difficulty }) {
  const w = await wasmExports();
  const enc = new TextEncoder();
  const write = (s) => {
    const b = enc.encode(s);
    const ptr = w.__wbindgen_export_0(b.length, 1);
    new Uint8Array(w.memory.buffer).set(b, ptr);
    return [ptr, b.length];
  };
  const [cp, cl] = write(challenge);
  const [pp, pl] = write(`${salt}_${expire_at}_`);
  const retptr = w.__wbindgen_add_to_stack_pointer(-16);
  try {
    w.wasm_solve(retptr, cp, cl, pp, pl, difficulty);
    const v = new DataView(w.memory.buffer);
    if (v.getInt32(retptr, true) === 0) {
      throw new Error("DeepSeek PoW 求解失败（超过难度上限，挑战可能已过期）");
    }
    return v.getFloat64(retptr + 8, true);
  } finally {
    w.__wbindgen_add_to_stack_pointer(16);
  }
}

// 组装 x-ds-pow-response 请求头的值：base64(JSON)
export async function powResponseHeader(ch) {
  const answer = await solvePow(ch);
  const payload = {
    algorithm: ch.algorithm,
    challenge: ch.challenge,
    salt: ch.salt,
    answer,
    signature: ch.signature,
    target_path: ch.target_path,
  };
  const json = JSON.stringify(payload);
  if (typeof Buffer !== "undefined") return Buffer.from(json, "utf8").toString("base64");
  return btoa(String.fromCharCode(...new TextEncoder().encode(json)));
}
