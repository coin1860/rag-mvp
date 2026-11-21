"""FastAPI layer for the local RAG MVP: one HTML page, three JSON endpoints.

Rag and Store are module-level singletons initialised lazily on first request,
so importing the module does no work (no model load, no DB open).
"""

from __future__ import annotations

import html
from functools import lru_cache

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
    store = get_store()
    if deep:
        ok, detail = get_rag().llm.reachable()
        gateway = {"ok": ok, "detail": detail}
    else:
        gateway = {"ok": None, "detail": "not probed (pass ?deep=1)"}
    return {
        "ready": store.size > 0,
        "chunks": store.size,
        "documents": store.documents(),
        "gateway": gateway,
        "model": settings.model,
    }


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>本地 RAG</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; box-sizing: border-box; min-height: 4rem; font-size: 1rem; }
  button { margin-top: .5rem; font-size: 1rem; padding: .4rem 1.2rem; }
  #result { margin-top: 1.5rem; }
  .card { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
  .card.refused { border-color: #c77; background: #fff7f5; }
  .src { font-size: .85rem; color: #555; margin-top: .6rem; border-top: 1px dashed #ddd; padding-top: .6rem; }
  .err { color: #b00; }
</style>
</head>
<body>
<h1>本地 RAG 知识库</h1>
<textarea id="q" placeholder="输入问题，例如：退款多久到账？"></textarea>
<button id="ask">提问</button>
<div id="result"></div>
<script>
const esc = s => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
document.getElementById('ask').onclick = async () => {
  const area = document.getElementById('result');
  area.innerHTML = '<p>思考中…</p>';
  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: document.getElementById('q').value}),
    });
    const data = await resp.json();
    if (!resp.ok) { area.innerHTML = '<p class="err">错误 ' + resp.status + ': ' + esc(data.detail || '') + '</p>'; return; }
    const cls = data.refused ? 'card refused' : 'card';
    const srcs = data.sources.map(s =>
      '<p>[' + s.n + '] ' + esc(s.path) + (s.heading ? ' #' + esc(s.heading) : '') +
      ' — score ' + s.score.toFixed(3) + '<br>' + esc(s.snippet) + '</p>').join('');
    area.innerHTML =
      '<div class="' + cls + '"><strong>' + (data.refused ? '已拒绝' : '回答') +
      ' (top_score ' + data.top_score.toFixed(3) + ')</strong><p>' + esc(data.answer) + '</p>' +
      (srcs ? '<div class="src"><strong>来源：</strong>' + srcs + '</div>' : '') + '</div>';
  } catch (e) {
    area.innerHTML = '<p class="err">请求失败: ' + esc(String(e)) + '</p>';
  }
};
</script>
</body>
</html>"""
