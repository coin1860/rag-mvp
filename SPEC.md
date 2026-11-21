# SPEC — 本地 RAG 知识库 MVP

状态：已确认（grill 阶段产出，2026-11-21）

## 目标

本地跑起来的最小可用 RAG：把 `docs/` 里的 Markdown 建成向量索引，用自然语言提问，返回**带引用来源**的答案。回答由 `agent2api` 网关的 `deepseek-v4.1-flash` 生成，向量由本地 fastembed 生成。

## 需求（可验收）

| ID | 需求 | 验收方式 |
|---|---|---|
| R1 | 索引 `docs/**/*.md`，按 Markdown 标题切块 | `POST /reindex` 返回 `documents`/`chunks` 计数 |
| R2 | 提问返回答案 + 至少一条来源（文件路径 + 命中片段） | `POST /ask` 响应里 `sources[].path` 命中预期文档 |
| R3 | 检索不到相关内容时**不编造**，明确说"知识库里没有" | 问一个库里完全没有的问题，`refused=true` 且未调用 LLM |
| R4 | 答案里的 `[1][2]` 标记与 `sources` 顺序一致 | 断言答案中出现的最大编号 ≤ `len(sources)` |
| R5 | 极简网页可提问并显示答案与来源 | 浏览器打开 `/`，输入问题，看到答案卡片 |
| R6 | 全部本地可复现，不依赖除网关外的任何外部服务 | `scripts/verify.py` 从空库起全新进程跑通 |

## 非目标（明确不做）

- 不做 PDF/OCR、不做增量文件监听、不做多轮对话记忆、不做 rerank 模型、不做鉴权、不做向量数据库（sqlite + numpy 够）。
- 不做流式输出。

## 关键决策与理由

1. **Embedding = 本地 fastembed `BAAI/bge-small-zh-v1.5`（512 维）**
   理由：`agent2api` 网关实测无 `/v1/embeddings`（404）；本地模型零 key、离线、67MB。中文检索质量足够 MVP。
   代价：首次使用要下载 67MB 模型；换模型必须重建索引（向量不可比）。
2. **生成 = `agent2api` 网关 `deepseek-v4.1-flash`**
   理由：免费额度、OpenAI 兼容、1M 上下文。用 `urllib` 直连，不引 openai SDK。
3. **检索 = 余弦相似度 + sqlite 存原文 + numpy 存向量**
   理由：MVP 规模（<1 万块）暴力检索毫秒级，省掉一个向量库依赖。
   归一化在写入时做一次，查询向量也归一化，点积即余弦。
4. **拒绝阈值 `min_score=0.35`，低于阈值直接返回 `refused=true`**
   理由：R3。宁可说不知道，也不要拿 LLM 的幻觉填空。阈值是经验值，写进 config 可调。
5. **答案必须带编号引用，引用编号由检索结果顺序决定**
   理由：R4。让"这句话从哪来"可核查，这是 RAG 唯一比直接问模型强的地方。

## 风险

- bge-small-zh 对纯英文文档召回弱于中文（示例库里放了一篇英文文档专门测这个）。
- 网关不可用 → `/ask` 返回 502 并说明是网关问题，不是 RAG 逻辑问题。

---

## 修订 1（实测后）— 拒绝判定改为两段式

初版写「阈值 `min_score=0.35`」是**没有数据支撑的猜测**，实测后推翻：

| 问题 | top 余弦 | 是否可答 |
|---|---|---|
| 退款多久到账 | 0.719 | 可答 |
| 客服几点上班 | 0.554 | 可答 |
| 导出支持哪些格式 | 0.596 | 可答 |
| How much does the Pro plan cost | **0.483** | 可答 |
| what is the capital of France | 0.439 | 不可答 |
| 今天北京天气怎么样 | 0.388 | 不可答 |
| 火星上的重力加速度 | 0.344 | 不可答 |
| **如何给 PDF 加密** | **0.523** | **不可答** |

结论：**没有单一余弦阈值能分开这两组**。0.523 的不可答问题压在 0.483 的可答问题之上；设 0.45 会漏编造，设 0.55 会杀掉英文召回。

改为两段式：
1. `min_score = 0.40` 只做粗筛，低谷直接拒，**不调用 LLM**（省钱、确定性、可单测）
2. 过了粗筛后，system prompt 要求模型在"材料不足以回答"时只输出哨兵 `NO_CONTEXT`；命中哨兵即 `refused=True`，且不返回 sources

已回归覆盖：`如何给 PDF 加密` 作为泄漏用例进了 `scripts/verify.py`。

开放决策（不在 MVP 范围）：真正的解法是 rerank 模型或更强的中英 embedder，代价是模型体积与首启延迟。

---

## 修订 2（interrogate 两名对抗者实测后）

改动全部由"重跑能复现"驱动，没有一条来自直觉：

| 问题 | 证据 | 处置 |
|---|---|---|
| 一个 sqlite 连接被 FastAPI 线程池共享 | 40 并发 `/health` → 23 个 500，`ProgrammingError: SQLite objects created in a thread...` | `check_same_thread=False` + `RLock` 串行化 |
| `overlap >= max_chars` 时窗口不前进 | `_windows("x"*300, 100, 200)` 永不返回 | 钳制 `overlap <= max_chars-1` |
| 代码块里的 `#` 被当成标题 | `# 注释不是标题` 成为 heading | 加围栏状态机 |
| `NO_CONTEXT` 用子串匹配 | 引用规则本身的好答案被误拒 | 改为行首 token 匹配 |
| 引用校验空集真空通过 | `cited=[]` 时 `all(...)` 恒真，一个从不引用的应用也能过 | verify 增加「非拒绝必须有 ≥1 引用」 |
| **查询前缀有害** | 去掉后每条可答题都涨分：客服 0.554→0.665、英文题 0.483→0.606，且 top-1 变正确 | 关闭 `QUERY_PREFIX` |
| **粗筛阈值同时过滤了证据** | 「团队版有没有单点登录」可答却被拒：答案块 0.482 < 阈值 0.50 | 粗筛只管范围，证据集给全 top-k |
| **prompt 只用 180 字摘要** | 答案句在第二段，被截断 → 模型合理拒答 | 模型看整块，摘要仅供 UI |
| 阈值缺少数据 | 见修订 1 标定表，重测后 `min_score` 0.40→0.50，`top_k` 4→6 | 由实测决定 |

**已知限制（不在 MVP 内解决，见 TICKETS T8）**：跨语言术语鸿沟。中文问「团队版有没有单点登录」，`pricing.md` 写的是 `single sign-on`，答案块排 **10/22**，`top_k=6` 取不到 → 模型只能拒答。解法是多语言 embedder（`bge-m3`，1024 维）或混合字面检索，不是调大 `top_k`（那等于把一半语料塞进 prompt）。

---

## 修订 3（双轴 code review 之后）

`/skill:code-review` 两轴（Standards + Spec）各报 6 条，逐条处置：

| 轴 | 发现 | 处置 |
|---|---|---|
| Standards | `doc_hash` 声明 NOT NULL 却永远写 `""` | 真算：`sha256(文件内容)[:12]`，为将来的增量索引准备 |
| Standards | `Rag.__init__` 四个 `x if x is not None else settings.x` 样板 + 参数簇 | 只留 `top_k`/`min_score` 可注入，块大小回 `Settings` |
| Standards | `PAGE` 内嵌在 `app.py`，改前端与改路由同文件 | 移到 `web/index.html` |
| Standards | `sources` 与 `hits` 靠 `n-1` 下标算术重连 | prompt 直接从 `hits` 生成 |
| Standards | `chunker` 里两次链式三元重赋值 `cut` | 抽 `_sentence_cut(para, max_chars)`，不变量写进 docstring |
| Standards | `Embedder.dim` 是 property、`Store.dim()` 是方法 | 统一为 property |
| Spec | `/health` 的"模型是否就绪"用索引块数代理 | 新增 `embedder: {model, loaded, dim}`，且不触发加载 |
| Spec | `doc_hash` 从未被计算 | 同上（两轴独立命中同一处） |
| Spec | `embedder.py` 模块 docstring 声称加查询前缀，代码是 `""` | docstring 改成代码事实 |
| Spec | verify skill 写 `0.40` / `top 4`，出厂是 `0.50` / `6` | 同步，并补上 `doc_hash` 说明 |

**保留的超范围项（review 判为 scope creep，判定为有意为之，记录在此）**：
- `.github/workflows/ci.yml` —— 只跑单测，不需要网关；不引入外部依赖
- `.agents/skills/verify-rag/` —— SPEC 只要求 `scripts/verify.py`，但 agent 读的是 skill，脚本是它的执行体，两者配套才有用
- `?deep=1`、`/reindex` 的 `seconds`/`paths`、source 的 `score`/`n` —— 调试与证据所需，成本为零
