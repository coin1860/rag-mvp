"""Local embedding model: fastembed (ONNX, CPU, no API key).

Documents and queries are embedded the same way: no instruction prefix. bge-zh
documents one for queries, but it was ablated here and lowered every score
(see QUERY_PREFIX), so both paths are symmetric. Both go through this class so
the choice cannot drift between indexing and querying.
"""

from __future__ import annotations

import numpy as np

# bge-*-zh-* documents this prefix for short queries. Ablated on this corpus it
# lowered every score and demoted a correct top hit (客服几点上班: 0.554 -> 0.665
# without it; the English question 0.483 -> 0.606), so it is off. Re-measure
# before turning it back on: it is model- and corpus-dependent, not universal.
QUERY_PREFIX = ""


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed(["dimension probe"]).shape[1])
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = np.asarray(list(self._load().embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + query])
