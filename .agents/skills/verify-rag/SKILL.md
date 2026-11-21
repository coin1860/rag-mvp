---
name: verify-rag
description: "Launch, doctor, drive and prove the local RAG knowledge base in this repo (FastAPI service + one HTML page, sqlite index, local fastembed vectors, answers from an OpenAI-compatible gateway). Use when a change to retrieval, chunking, the refusal gate, or the prompt needs runtime evidence instead of a passing unit test."
---

# Verify: local RAG knowledge base

Drive the running service the way a user does: ask a real question over HTTP,
read the answer **and** the source it cites. A green `pytest` run proves neither
retrieval nor refusal — the bug that motivated this skill was an app whose unit
tests passed and whose end-to-end script could never terminate.

Everything below runs from the repo root, with `.venv/bin/python` and
`PYTHONPATH=.`.

## Launch

```bash
PYTHONPATH=. .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port <free-port>
```

- Pick the port by binding port 0 then closing it. **Never reuse a port that is
  already listening**: an older process still serving the previous code answers
  `/health` happily, and you then "prove" the fix against stale code. This is the
  single most expensive mistake in this repo's history.
- Ready means `/health` returns 200. It does **not** mean the index is populated:
  a fresh process legitimately reports `chunks: 0`. Requiring `chunks > 0` before
  `POST /reindex` deadlocks every run.
- Cold start loads the ONNX embedding model (~1–3 s warm, ~60 s if it must
  download 67 MB). Allow 120 s.
- Teardown: terminate the pid you started. Never kill by process name.

## Doctor

```bash
curl -s "127.0.0.1:<port>/health"                 # process + index state, no gateway call
curl -s "127.0.0.1:<port>/health?deep=1" | jq .gateway   # one real gateway probe
```

`/health` is deliberately cheap: a readiness poll must not cost a completion.
Only `?deep=1` talks to the gateway. If `gateway.ok` is false, the failure is the
gateway, not retrieval — say so instead of debugging the RAG.

## Drive

```bash
curl -s -XPOST "127.0.0.1:<port>/reindex"                      # builds the index, returns counts
curl -s -XPOST "127.0.0.1:<port>/ask" -H 'Content-Type: application/json' \
     -d '{"question":"退款多久到账"}'
```

Better: run the whole sequence as one command, which is what CI and CI-minded
humans should use.

```bash
PYTHONPATH=. .venv/bin/python scripts/verify.py     # exits 0 only if every check passes
```

Always check the page, even when the change is not front-end: `GET /` must
return `content-type: text/html`. **A route that returns a bare `str` in FastAPI
emits `application/json`**, and the browser then displays the source text while
every status-code check still passes. This shipped once. Then drive one question
through the textarea and read the rendered card, not the console.

## Evidence

Capture, per question: `refused`, `top_score`, the answer text, and every
`path#heading`. The proof standards for this repo:

- Exercise `POST /ask`, never `Rag.ask` directly — the bug class lives in the
  wiring (empty index, cached store, model not loaded), not in the pure function.
- Assert **both** directions: an answerable question returns `refused: false`
  with the expected source path, and an unanswerable one returns `refused: true`
  with `sources: []`.
- Assert every `[n]` in the answer satisfies `n <= len(sources)`.
- A refusal without a printed `top_score` is not evidence; the score is how you
  tell "the gate fired" from "the model refused".
- Keep artifacts: the script prints its transcript; paste it into the PR rather
  than a summary of it.

## Cleanup

`scripts/verify.py` terminates its own subprocess in a `finally` block, even when
launch failed. If you launched by hand, `kill <pid>` that pid. Leave
`data/index.db` and the model cache in place — they are inputs, not scratch.

## Gotchas

- `data/index.db` is shared by every instance. Two `/reindex` calls at once can
  interleave; drive one instance at a time.
- Changing `RAG_EMBED_MODEL` invalidates every stored vector. The next run must
  `POST /reindex` or `Store.search` raises a dimension-mismatch error.
- Cosine similarity ranks; it does not adjudicate. Measured on this corpus, an
  unanswerable question (`如何给 PDF 加密`) scored 0.523, above an answerable
  English one at 0.483. That is why refusal is two-stage — see `SPEC.md` 修订 1.
  Do not "tighten the threshold" to fix a refusal bug without re-measuring both
  groups; you will break the other side.
- Reading the page's HTML with `TestClient` cannot tell you whether a browser
  renders it. Check the header, then look at it.
- A hand-run launch strands a process when a later step fails: wrap the whole
  sequence in `trap 'kill $PID' EXIT` or just run `scripts/verify.py`, which does
  that in a `finally`. This was observed while proving this skill, not theorised.
- The gateway is a free shared quota: keep LLM-touching checks to a handful per
  run, and expect a 502 with a clear `detail` when it is rate-limited.

## Feature map

`features/README.md` lists the user-facing journeys this skill covers.
