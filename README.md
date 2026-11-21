# rag-mvp

本地 RAG 知识库 MVP：把 `docs/` 里的 Markdown 建成向量索引，用自然语言提问，
返回**带引用来源**的答案；材料不足以回答时明确拒答，不编造。

- 向量：本地 `fastembed` + `BAAI/bge-small-zh-v1.5`（512 维，无需 API key，离线）
- 生成：任意 OpenAI 兼容端点（默认指向本机网关的 `deepseek-v4.1-flash`）
- 存储：sqlite 存原文 + numpy 存向量，暴力余弦检索
- 界面：单文件 HTML，无构建步骤

## 快速开始

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env          # 填 RAG_API_KEY

PYTHONPATH=. .venv/bin/python -m uvicorn app:app --port 8765
# 打开 http://127.0.0.1:8765 ，或者
curl -s -XPOST localhost:8765/reindex
curl -s -XPOST localhost:8765/ask -H 'Content-Type: application/json' \
     -d '{"question":"退款多久到账"}'
```

首次运行会下载 67MB 的 embedding 模型，之后离线可用。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 单页界面 |
| `POST` | `/ask` | `{"question": str}` → `{answer, refused, top_score, sources[]}` |
| `POST` | `/reindex` | 重建 `docs/**/*.md` 索引，返回文档数/块数 |
| `GET` | `/health` | 索引状态；`?deep=1` 才做一次网关探测 |

## 验证

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q     # 19 个单测，无网络
PYTHONPATH=. .venv/bin/python scripts/verify.py      # 25 项端到端，需要网关
```

`scripts/verify.py` 自己起一个新进程、随机空闲端口，跑完关掉，只在全部通过时返回 0。
它存在的理由是：**单测通过证明不了检索正确，也证明不了页面能渲染**。

项目内的验证手册在 `.agents/skills/verify-rag/`（pstack 的 verification skill 格式）。

## 关键设计

- **拒答是两段式**：余弦 `min_score=0.50` 只做范围粗筛（低谷直接拒，不调模型）；
  过了粗筛由模型在材料不足时输出哨兵 `NO_CONTEXT`。
  单一阈值做不到——实测不可答的问题余弦 0.595，压在可答的 0.561 之上（见 `SPEC.md 修订 1`）。
- **粗筛不过滤证据**：过闸后把全部 top-k 交给模型。曾因共用阈值把答案块删掉，
  导致可答的问题被拒（`SPEC.md 修订 2`）。
- **prompt 用整块文本，摘要只给界面**：180 字截断曾把答案句切掉。
- **查询不加 bge 指令前缀**：实测反而降分（客服题 0.554→0.665）。

## 已知限制

- **跨语言术语鸿沟**：中文问「单点登录」，文档写 `single sign-on`，答案块排 10/22，
  `top_k=6` 取不到 → 拒答。解法是多语言 embedder 或混合字面检索，见 `TICKETS.md T8`。
- 不做增量索引、不做 rerank、不做鉴权、不做流式输出、只支持 Markdown。
- `data/index.db` 被多实例共享，同一时刻只驱动一个实例。

## 文档

- `SPEC.md` — 需求、验收方式、三次修订的实测数据
- `TICKETS.md` — 可验证单元与未做项（T8）
- `.agents/skills/verify-rag/` — 怎么真跑一遍并拿到证据
