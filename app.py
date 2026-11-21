"""FastAPI layer for the local RAG MVP: one HTML page, three JSON endpoints.

Rag and Store are module-level singletons initialised lazily on first request,
so importing the module does no work (no model load, no DB open).
"""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag.config import settings
from rag.llm import LLMError
from rag.pipeline import Rag
from rag.store import Store


@lru_cache(maxsize=1)
def get_rag() -> Rag:
    return Rag()


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store(settings.db_path)


app = FastAPI(title="local-rag-mvp")


class AskBody(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    # Returning a bare str makes FastAPI emit application/json and the browser
    # shows the source text. The response class is part of the contract.
    return HTMLResponse(PAGE)


@app.post("/ask")
def ask(body: AskBody) -> dict:
    try:
        answer = get_rag().ask(body.question)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM gateway error: {exc}") from exc
    return {
        "answer": answer.text,
        "refused": answer.refused,
        "top_score": answer.top_score,
        "sources": [
            {
                "n": s.n,
                "path": s.path,
                "heading": s.heading,
                "score": s.score,
                "snippet": s.snippet,
            }
            for s in answer.sources
        ],
    }


@app.post("/reindex")
def reindex() -> dict:
    stats = get_rag().index_documents()
    return {
        "documents": stats.documents,
        "chunks": stats.chunks,
        "seconds": stats.seconds,
        "paths": stats.paths,
    }


@app.get("/health")
def health(deep: int = 0) -> dict:
    """Cheap by default: a readiness poll must never cost a gateway request.

    `?deep=1` runs the one-call gateway probe, for a single explicit check.
    """
    rag = get_rag()
    store = rag.store
    if deep:
        ok, detail = rag.llm.reachable()
        gateway = {"ok": ok, "detail": detail}
    else:
        gateway = {"ok": None, "detail": "not probed (pass ?deep=1)"}
    embedder = rag.embedder
    return {
        "ready": store.size > 0,
        "chunks": store.size,
        "documents": store.documents(),
        "gateway": gateway,
        # "is the model ready" must answer for the embedding model too, not just
        # name the generation model. Loading it here would make /health expensive,
        # so report whether it is already loaded and its width once it is.
        "embedder": {
            "model": embedder.model_name,
            "loaded": embedder.loaded,
            "dim": embedder.dim if embedder.loaded else store.dim,
        },
        "model": settings.model,
    }


PAGE = (Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")
