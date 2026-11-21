import numpy as np
import pytest

from rag.pipeline import Rag
from rag.store import Row, Store


class FakeEmbedder:
    """Deterministic embeddings: each vocabulary token maps to its own axis."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.dim = len(vocab)

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.split():
            if tok in self.vocab:
                v[self.vocab[tok]] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return "FAKE-ANSWER [1]"


VOCAB = {"refund": 0, "arrives": 1, "days": 2, "account": 3, "delete": 4}


def _rag(tmp_path, **kw):
    store = Store(tmp_path / "db")
    embedder = FakeEmbedder(VOCAB)
    llm = FakeLLM()
    opts = {"min_score": 0.3}
    opts.update(kw)
    rag = Rag(store=store, embedder=embedder, llm=llm, **opts)
    return rag, store, llm


def _seed(store, rows_texts):
    store.reset()
    store.add(
        [Row(id=i + 1, doc_path=p, heading=h, text=t) for i, (p, h, t) in enumerate(rows_texts)],
        EMBEDDER.embed_documents([t for _, _, t in rows_texts]),
    )


EMBEDDER = FakeEmbedder(VOCAB)


def test_index_documents_counts(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nrefund arrives in days.\n" * 5, encoding="utf-8")
    (docs / "b.md").write_text("no heading text about account delete.\n", encoding="utf-8")
    rag, _, _ = _rag(tmp_path)
    stats = rag.index_documents(docs)
    assert stats.documents == 2
    assert stats.chunks == 6
    assert stats.paths == ["a.md", "b.md"]
    assert rag.store.size == 6


def test_ask_returns_sources_and_prompt_contains_reference(tmp_path):
    rag, store, llm = _rag(tmp_path)
    _seed(store, [("refund-policy.md", "退款", "refund arrives in days")])
    ans = rag.ask("refund arrives")
    assert ans.refused is False
    assert ans.sources[0].path == "refund-policy.md"
    assert len(llm.calls) == 1
    prompt = llm.calls[0][-1]["content"]
    assert "[1]" in prompt
    assert "refund-policy.md" in prompt


def test_ask_below_threshold_refuses_without_calling_llm(tmp_path):
    rag, store, llm = _rag(tmp_path, min_score=0.99)
    _seed(store, [("refund-policy.md", "退款", "refund arrives in days")])
    ans = rag.ask("account delete")
    assert ans.refused is True
    assert llm.calls == []


def test_ask_on_empty_index_refuses(tmp_path):
    rag, _, llm = _rag(tmp_path)
    ans = rag.ask("refund arrives")
    assert ans.refused is True
    assert ans.sources == []
    assert llm.calls == []
