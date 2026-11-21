import numpy as np
import pytest

from rag.store import Row, Store

DIM = 8


def _orthogonal(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _store(tmp_path):
    s = Store(tmp_path / "test.db")
    s.reset()
    s.add(
        [Row(id=i + 1, doc_path=f"d{i}.md", heading=f"h{i}", text=f"doc {i}") for i in range(3)],
        np.stack([_orthogonal(i) for i in range(3)]),
    )
    return s


def test_search_returns_matching_row_first_with_score_one(tmp_path):
    s = _store(tmp_path)
    hits = s.search(_orthogonal(1), k=3)
    assert hits[0].doc_path == "d1.md"
    assert hits[0].text == "doc 1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)


def test_search_on_empty_index_returns_empty_list(tmp_path):
    s = Store(tmp_path / "empty.db")
    assert s.search(_orthogonal(0)) == []
    assert s.size == 0


def test_dimension_mismatch_raises_valueerror(tmp_path):
    s = _store(tmp_path)
    bad = np.zeros(DIM + 4, dtype=np.float32)
    bad[0] = 1.0
    with pytest.raises(ValueError):
        s.search(bad)


def test_reset_empties_the_store(tmp_path):
    s = _store(tmp_path)
    assert s.size == 3
    assert s.dim() == DIM
    s.reset()
    assert s.size == 0
    assert s.documents() == []
    assert s.search(_orthogonal(0)) == []
