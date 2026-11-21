"""What /health must answer, and what the index must record."""

from fastapi.testclient import TestClient

import app as app_module
from rag.pipeline import Rag


def test_health_reports_embedder_state_not_just_a_model_name(tmp_path):
    rag = Rag()
    rag.store.reset()
    body = TestClient(app_module.app).get("/health").json()
    assert body["embedder"]["model"], "must name the embedding model"
    assert "loaded" in body["embedder"], "must say whether it is loaded"


def test_index_records_a_content_hash_per_document(tmp_path):
    (tmp_path / "a.md").write_text("# 标题\n\n正文一。\n", encoding="utf-8")
    rag = Rag()
    rag.index_documents(tmp_path)
    hashes = {
        row[0]
        for row in rag.store._conn.execute("SELECT DISTINCT doc_hash FROM chunks")  # noqa: SLF001
    }
    assert len(hashes) == 1
    value = hashes.pop()
    assert len(value) == 12 and int(value, 16) >= 0, value

    before = value
    (tmp_path / "a.md").write_text("# 标题\n\n正文二。\n", encoding="utf-8")
    rag.index_documents(tmp_path)
    after = rag.store._conn.execute("SELECT DISTINCT doc_hash FROM chunks").fetchone()[0]  # noqa: SLF001
    assert after != before, "a content change must change the hash"
