# magpie-deepseek-web

把 **DeepSeek 网页版**（chat.deepseek.com）代理成 magpie 的一个供应商：在 magpie 里登录一次，所有智能体就能像调用普通 API 一样用网页版的模型，多账号、故障切换、用量显示都由 magpie 接管。

## 安装

```shell
magpie plugin add /path/to/magpie-deepseek-web
magpie plugin login deepseek-web
```

应用里：插件 → 添加插件 → 填入本文件夹路径 → 在 DeepSeek 网页版那一行点登录。

## 登录（获取 userToken）

插件用的是网页版自己的会话 token，不是官方 API key：

1. 浏览器打开 [chat.deepseek.com](https://chat.deepseek.com) 并登录；
2. 按 F12 打开开发者工具，进「控制台」，执行：

   ```js
   JSON.parse(localStorage.getItem("userToken")).value
   ```

3. 复制输出的那串 token（64 位），粘贴进 magpie 的登录框。插件会当场校验（读取账号信息），通过后以邮箱/手机号命名账号。

token 等同于你的账号登录凭据，只保存在本机 magpie 的 `plugin-auth.json`（权限 600）里；只发往 `chat.deepseek.com`。可以登录多个账号做故障切换。网页版 token 没有公开的有效期，失效时账号会被标记，重新粘贴一次即可。

## 模型

| 模型 | 说明 |
| --- | --- |
| `deepseek-web/deepseek-chat` | 网页版快速模式（V3，思考关闭） |
| `deepseek-web/deepseek-reasoner` | 网页版深度思考（思考开启，输出 `reasoning_content`） |

两者都按免费（`free: true`）声明，不占套餐额度统计。

## 工作原理

网页版没有公开 API，插件直接走它的网页协议（2026-10 实测）：

1. `POST /api/v0/chat_session/create` 建一个临时会话；
2. `POST /api/v0/chat/create_pow_challenge` 拿 PoW 挑战，用 DeepSeek 官方网页自带的 `sha3_wasm_bg.wasm`（已随插件打包）在本地解出 nonce，组装 `x-ds-pow-response` 头；
3. `POST /api/v0/chat/completion`（SSE）发出 prompt，把网页版的分片事件流（THINK/RESPONSE fragment 补丁）转成 OpenAI chat completions 的流式增量；
4. 回答结束后删除临时会话，不在你的网页版历史里留痕。

鉴权只需要 `Authorization: Bearer <userToken>`；PoW 是唯一的服务端强制校验。

### 多轮对话怎么处理

网页版接口只接受一条 `prompt`（服务端会话不在插件手里），所以插件把 OpenAI 的 `messages` 拍平：系统提示放在开头，多轮对话转写成带角色标记的记录，最后一条用户消息单独点名，要求模型以助手身份直接回复。单条用户消息则原样发送。

### 工具调用怎么处理（文本协议桥接）

网页版模型没有原生的工具调用能力，插件做了一层桥接：

- 请求里带 `tools` 时，插件把工具定义（名称、描述、参数 JSON Schema）写进 prompt，并教模型用固定格式输出调用：

  ```
  <tool_call>
  {"name": "bash", "arguments": {"command": "ls -la"}}
  </tool_call>
  ```

- 回答流里出现的 `<tool_call>…</tool_call>` 块被实时解析成 OpenAI 的 `tool_calls` 结构（`finish_reason: "tool_calls"`），智能体就能真正执行工具；其余文本照常作为正文输出。
- 解析器容忍任意分包和模型把块包进 ```` ``` ```` 围栏的坏习惯；JSON 解析失败时原文降级为正文，不会吞内容。
- prompt 里同时明确了铁律：**严禁编造工具的返回结果**——早期版本没给模型调用通道，模型曾照着对话记录的格式自己"表演"了整个工具循环（连命令输出都是编的），这个坑已经堵上。

## 限制

- **工具调用是桥接的，不是原生的**：模型偶尔会输出不合协议的调用块（该次调用降级为正文），复杂智能体任务的成功率低于原生工具调用模型，跑偏时重试通常即可。
- **不支持图片输入**：消息里的图片会被替换成占位说明。
- **上下文**：按 128K 声明（网页版单条输入上限 260 万字符，足够宽松，但超长对话被拍平后效果受模型本身上下文约束）。
- **速率**：受 DeepSeek 网页版自身的频率限制，每个账号并发默认 3（`magpie.maxConcurrency`），被限流时 magpie 会按厂商回答做故障切换。
- **协议风险**：非官方接口，DeepSeek 改版（尤其是 PoW 算法或 wasm 文件名）后需要更新插件。

## 选项

在 `magpie plugin options magpie-deepseek-web '<json>'`（或应用里插件那一行的选项按钮）设置：

```json
{ "keepSessions": false, "search": false }
```

- `keepSessions`：`true` 时保留每次调用的会话（能在网页版历史里看到），默认用完即删。
- `search`：`true` 时允许网页版联网搜索（搜索结果的引用会混进正文），默认关闭。

## 文件

| 文件 | 作用 |
| --- | --- |
| `index.mjs` | 插件入口：供应商声明、登录、请求流程 |
| `lib.mjs` | 纯函数：messages 拍平（含工具协议）、SSE 事件流 → OpenAI chunk 转换、`<tool_call>` 流式提取 |
| `pow.mjs` | PoW 求解器（加载并调用官方 wasm） |
| `sha3_wasm_bg.wasm` | DeepSeek 网页版自带的 sha3 wasm（`fe-static.deepseek.com` 原样拷贝） |

测试过的环境：magpie 0.1.2283（插件宿主 Bun 1.4.4）、`magpie provider test`、沙箱网关 `magpie web --gateway` 的流式/非流式/多轮/深度思考请求，以及工具调用两轮往返（发起调用 → 回填结果 → 最终答复）。
