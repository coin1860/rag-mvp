"""Local embedding model: fastembed (ONNX, CPU, no API key).

Documents are embedded as-is; queries get the bge Chinese instruction prefix,
which the model was trained with. Getting this backwards silently degrades
recall, so both paths go through this class and nowhere else.
"""

from __future__ import annotations

import numpy as np

# bge-*-zh-* recommends this prefix for short queries, never for passages.
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


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
