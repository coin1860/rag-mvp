"""Regression guards for failures an adversary found by running the code."""

import threading

import numpy as np
import pytest

from rag.chunker import chunk_markdown
from rag.pipeline import NO_CONTEXT, Rag
from rag.store import Row, Store


def test_overlap_at_or_above_max_chars_terminates():
    # A window that does not advance used to loop forever inside /reindex.
    chunks = chunk_markdown("x" * 300, max_chars=100, overlap=200)
    assert chunks and all(len(c.text) <= 100 for c in chunks)


def test_hash_inside_a_code_fence_is_not_a_heading():
    doc = "# 标题\n\n```python\n# 注释不是标题\nprint(1)\n```\n\n正文。\n"
    headings = [c.heading for c in chunk_markdown(doc)]
    assert headings == ["标题"], headings


def test_sentinel_matched_only_as_a_leading_token():
    class EchoLLM:
        def __init__(self, text):
            self.text = text

        def chat(self, messages, **kwargs):
            return self.text

    class FixedEmbedder:
        dim = 4

        def embed_documents(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

        def embed_query(self, q):
            return np.ones((1, 4), dtype=np.float32) / 2

    store = Store(":memory:")
    store.add([Row(id=1, doc_path="a.md", heading="h", text="退款 3 个工作日到账")], np.ones((1, 4), np.float32))

    rag = Rag(store=store, embedder=FixedEmbedder(), llm=EchoLLM(f"我会输出{NO_CONTEXT}，但答案是退款 3 个工作日到账 [1]"), min_score=0.1)
    assert rag.ask("退款").refused is False

    rag.llm = EchoLLM(NO_CONTEXT)
    assert rag.ask("退款").refused is True


def test_store_survives_concurrent_reads():
    # sqlite3 connections are thread-bound by default; one Store is shared by
    # FastAPI's threadpool, which used to raise ProgrammingError.
    store = Store(":memory:")
    store.add([Row(id=1, doc_path="a.md", heading="", text="x")], np.ones((1, 4), np.float32))
    errors: list[Exception] = []

    def read():
        try:
            store.size
            store.documents()
            store.search(np.ones(4, np.float32), k=1)
        except Exception as exc:  # noqa: BLE001 - the point is to catch anything
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
