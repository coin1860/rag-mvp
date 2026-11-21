"""Vector index: sqlite for text, numpy for vectors.

Vectors are L2-normalised on write and on query, so cosine similarity is a dot
product. Brute force over an in-memory matrix is milliseconds at MVP scale and
removes an entire dependency.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    doc_path  TEXT NOT NULL,
    doc_hash  TEXT NOT NULL,
    heading   TEXT NOT NULL,
    ord       INTEGER NOT NULL,
    text      TEXT NOT NULL,
    vec       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_path);
"""


@dataclass(frozen=True)
class Row:
    id: int
    doc_path: str
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc_path: str
    heading: str
    text: str
    score: float


class Store:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._matrix: np.ndarray | None = None
        self._rows: list[Row] = []

    # -- writes ---------------------------------------------------------
    def reset(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        self._matrix, self._rows = None, []

    def add(self, rows: list[Row], vecs: np.ndarray) -> None:
        assert len(rows) == len(vecs), "one vector per row"
        self._conn.executemany(
            "INSERT INTO chunks (doc_path, doc_hash, heading, ord, text, vec)"
            " VALUES (?,?,?,?,?,?)",
            [
                (r.doc_path, "", r.heading, i, r.text, vecs[i].astype(np.float32).tobytes())
                for i, r in enumerate(rows, start=1)
            ],
        )
        self._conn.commit()
        self._matrix, self._rows = None, []

    # -- reads ----------------------------------------------------------
    def _load(self) -> None:
        if self._matrix is not None:
            return
        cur = self._conn.execute("SELECT id, doc_path, heading, text, vec FROM chunks ORDER BY id")
        rows: list[Row] = []
        vecs: list[np.ndarray] = []
        for cid, path, heading, text, blob in cur:
            rows.append(Row(id=cid, doc_path=path, heading=heading, text=text))
            vecs.append(np.frombuffer(blob, dtype=np.float32))
        self._rows = rows
        self._matrix = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def documents(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT doc_path FROM chunks ORDER BY doc_path")
        return [r[0] for r in cur]

    def dim(self) -> int | None:
        self._load()
        return None if self._matrix is None or self._matrix.size == 0 else int(self._matrix.shape[1])

    def search(self, qvec: np.ndarray, k: int = 4) -> list[Hit]:
        self._load()
        if self._matrix is None or self._matrix.size == 0:
            return []
        q = qvec.reshape(-1).astype(np.float32)
        if q.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"dimension mismatch: query {q.shape[0]} vs index {self._matrix.shape[1]}"
                " — the embedding model changed, reindex required"
            )
        scores = self._matrix @ q
        order = np.argsort(-scores)[:k]
        return [
            Hit(
                doc_path=self._rows[i].doc_path,
                heading=self._rows[i].heading,
                text=self._rows[i].text,
                score=float(scores[i]),
            )
            for i in order
        ]
