# TICKETS — 每个 ticket 是一个可验证单元

约定：`Check:` 是这个 ticket 完成的**确切证据**，做完立刻验，不合格不许进下一个。

---

## T1 chunker：Markdown 标题感知切块
`Writes:` `rag/chunker.py` `tests/test_chunker.py`
- 按 `#`~`######` 分层；每个 chunk 带 heading 路径（`父 > 子`）
- 单块 ≤ `max_chars`（默认 600），相邻块重叠 `overlap`（默认 80）字符
- 空文档 → 0 块；无标题文档 → 单块，heading 为空

`Check:` `pytest tests/test_chunker.py -q` 全绿，且断言里包含"最长块 ≤ max_chars"和"标题出现在块的 heading 字段"

---

## T2 embedder：本地 fastembed 封装
`Writes:` `rag/embedder.py`
- 懒加载模型；`embed(texts)` → `np.ndarray (n, 512)` float32，已 L2 归一化
- 查询侧加 bge 中文指令前缀，文档侧不加
- 输出的 `dim` 可读，用于校验索引与模型是否匹配

`Check:` `python -c "from rag.embedder import Embedder; e=Embedder();v=e.embed(['退款政策']);print(v.shape, (v**2).sum())"` → `(1, 512) 1.0`

---

## T3 store：sqlite + numpy 的索引与检索
`Writes:` `rag/store.py` `tests/test_store.py`
- 表 `chunks(id, doc_path, doc_hash, heading, ord, text, vec BLOB)`
- 写入前清空重建（MVP 不做增量）；`doc_hash` 留着给以后的增量用
- `search(qvec, k)` → 按余弦排序的 `Hit` 列表（含 score）
- 索引为空时返回空列表而不是抛异常

`Check:` `pytest tests/test_store.py -q`；含一个"构造 3 个正交向量、查询命中正确那条"的断言

---

## T4 llm：网关客户端
`Writes:` `rag/llm.py`
- `urllib.request` POST `{base}/chat/completions`，OpenAI 兼容
- 缺 API key / 网络失败 / 非 200 → 抛 `LLMError`，消息里带状态码和响应体前 200 字符
- 超时 90s

`Check:` 单元测试用一个假 URL 断言抛 `LLMError`；真链路由 T7 覆盖

---

## T5 pipeline：index 与 ask
`Writes:` `rag/pipeline.py` `rag/config.py` `tests/test_pipeline.py`
- `index_documents(dir)` → `IndexStats(documents, chunks, seconds)`
- `ask(question, k)` → `Answer(text, sources, refused, top_score)`
- 低于 `min_score` → `refused=True` 且**不调用 LLM**（用 spy 断言）
- prompt 里每块前加 `[n] path#heading`，并明确要求"只依据材料，答不出就说不知道"

`Check:` `pytest tests/test_pipeline.py -q`，其中必须有一条断言"refused 时 LLM 未被调用"

---

## T6 app：FastAPI + 极简网页
`Writes:` `app.py`
- `GET /` 单页 HTML（无外部 CDN，无构建步骤）
- `POST /ask {question}` → `{answer, sources, refused, top_score}`
- `POST /reindex` → `IndexStats`
- `GET /health` → 模型是否就绪、索引块数、网关是否可达

`Check:` `curl -s localhost:PORT/health` 返回 `chunks>0`

---

## T7 verify：独立证据脚本（本项目最重要的一步）
`Writes:` `scripts/verify.py`
- **Launch**：随机空闲端口起**新** uvicorn 进程（绝不复用已在跑的端口）
- **Doctor**：轮询 `/health` 直到就绪，超时即失败
- **Drive**：`POST /reindex` → 问 3 个问题（中文精确可答 / 英文 / 库里没有）
- **Evidence**：断言 可答题 `refused=False` 且来源命中预期文件；不可答题 `refused=True`；打印每条证据
- **Cleanup**：只 kill 自己起的进程

`Check:` `python scripts/verify.py` 退出码 0，输出里能看到 `PASS` 行与实际来源路径

---

## 依赖顺序

T1 → T2 → T3 → T5（T4 可并行）→ T6 → T7

共享写：`rag/store.py` 与 `rag/pipeline.py` 不能并行改。

---

## T8（未做，v2）跨语言召回

`Check:` 中文问「团队版有没有单点登录」返回 `refused=False` 且来源含 `pricing.md#Team Plan`。
现状：答案块 rank 10/22，`top_k=6` 取不到，模型拒答。候选解法：换 `bge-m3`（1024 维、多语言、本地 onnx 或硅基流动免费 API）或加一层字面检索混合。**不要**用调大 `top_k` 解决。
