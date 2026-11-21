"""Index and ask: the only two operations this MVP has.

The retrieval-then-generate seam lives here, so the API layer stays dumb and the
tests can spy on the LLM.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_markdown
from .config import settings
from .embedder import Embedder
from .llm import LLM
from .store import Row, Store

NO_CONTEXT = "NO_CONTEXT"

# Cosine similarity ranks, it does not adjudicate: measured on this corpus, an
# unanswerable question ("如何给 PDF 加密") scored 0.523 above an answerable
# English one (0.483). So the score gate only catches the obvious misses, and
# the model makes the final call by emitting a sentinel instead of an answer.
SYSTEM = (
    "你是一个只依据给定材料回答问题的助手。规则：\n"
    "1. 只能用材料里出现的信息，不许补充常识或推测。\n"
    "2. 每个结论后面标注来源编号，如 [1]；允许多个来源 [1][2]。\n"
    f"3. 如果材料不足以回答问题（哪怕只沾到边），只输出 {NO_CONTEXT} 这一个词，不要输出任何解释。\n"
    "4. 用中文回答，简洁，不超过 120 字。"
)


@dataclass(frozen=True)
class Source:
    n: int
    path: str
    heading: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]
    refused: bool
    top_score: float


@dataclass(frozen=True)
class IndexStats:
    documents: int
    chunks: int
    seconds: float
    paths: list[str] = field(default_factory=list)


class Rag:
    def __init__(
        self,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        max_chars: int | None = None,
        overlap: int | None = None,
    ) -> None:
        self.store = store or Store(settings.db_path)
        self.embedder = embedder or Embedder(settings.embed_model)
        self.llm = llm or LLM(settings.base_url, settings.api_key, settings.model, settings.timeout)
        self.top_k = top_k if top_k is not None else settings.top_k
        self.min_score = min_score if min_score is not None else settings.min_score
        self.max_chars = max_chars if max_chars is not None else settings.max_chars
        self.overlap = overlap if overlap is not None else settings.overlap

    # -- index ----------------------------------------------------------
    def index_documents(self, docs_dir: Path | None = None) -> IndexStats:
        started = time.perf_counter()
        docs_dir = Path(docs_dir or settings.docs_dir)
        files = sorted(docs_dir.rglob("*.md"))
        self.store.reset()

        rows: list[Row] = []
        texts: list[str] = []
        for path in files:
            raw = path.read_text(encoding="utf-8")
            for chunk in chunk_markdown(raw, max_chars=self.max_chars, overlap=self.overlap):
                rows.append(
                    Row(
                        id=len(rows) + 1,
                        doc_path=str(path.relative_to(docs_dir)),
                        heading=chunk.heading,
                        text=chunk.text,
                    )
                )
                # The heading carries retrieval signal ("团队版 > 单点登录") that
                # a text-only vector cannot see; it stays out of the stored text
                # so the prompt still shows the raw passage.
                texts.append(f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text)

        if rows:
            self.store.add(rows, self.embedder.embed_documents(texts))

        return IndexStats(
            documents=len(files),
            chunks=len(rows),
            seconds=round(time.perf_counter() - started, 2),
            paths=[str(p.relative_to(docs_dir)) for p in files],
        )

    # -- ask ------------------------------------------------------------
    def ask(self, question: str, k: int | None = None) -> Answer:
        k = k or self.top_k
        hits = self.store.search(self.embedder.embed_query(question), k=k)
        if not hits:
            return Answer("索引是空的，先建索引。", [], True, 0.0)

        top = hits[0].score
        if top < self.min_score:
            return Answer(
                f"知识库里没有相关内容（最高相似度 {top:.3f}，低于阈值 {self.min_score:.2f}）。",
                [],
                True,
                top,
            )

        # The gate above decides whether the question is in scope at all; it must
        # NOT also filter the evidence. Reusing min_score here deleted the chunk
        # that actually answered the question ("团队版有没有单点登录": the Team
        # chunk scores 0.482, below the 0.50 gate, so the model was shown only
        # wrong material and correctly refused an answerable question).
        sources = [
            Source(n=i, path=h.doc_path, heading=h.heading, score=round(h.score, 3), snippet=h.text[:180])
            for i, h in enumerate(hits, start=1)
        ]
        # The model gets the whole passage; `snippet` (180 chars) exists only for
        # the UI. Truncating the evidence silently deleted the sentence that
        # answered the question: the Team SSO fact sits past char 180.
        material = "\n\n".join(
            f"[{s.n}] {s.path}" + (f"#{s.heading}" if s.heading else "") + f"\n{hits[s.n - 1].text}"
            for s in sources
        )
        text = self.llm.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"材料：\n{material}\n\n问题：{question}"},
            ]
        )
        # Match the sentinel as a leading token, not a substring: a grounded
        # answer that merely quotes the rule must not be discarded.
        if text.strip().upper().startswith(NO_CONTEXT):
            return Answer("知识库里没有相关内容（检索到的材料不足以回答这个问题）。", [], True, top)
        return Answer(text, sources, False, top)
