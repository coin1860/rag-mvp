# Feature map — local RAG

Index of the journeys `verify-rag` covers. One file per user-facing feature.
A proof that only drives `/ask` is incomplete: refusal and reindex change the
meaning of every answer, so each has its own journey.

| # | Feature | File | Why it needs runtime proof |
|---|---|---|---|
| 1 | Ask a question, get a cited answer | `ask.md` | retrieval + prompt + citation numbering all only exist at runtime |
| 2 | Refuse when the corpus does not answer | `refusal.md` | the gate is a score plus a model sentinel; neither is unit-testable end to end |
| 3 | Rebuild the index | `reindex.md` | a stale index or a desynced cached matrix looks exactly like a retrieval bug |

Covered by `scripts/verify.py`: 1 and 2 (including the 0.523 leak regression), plus
one cross-lingual question. 3 is covered only as a precondition — add a journey
file here if a change touches indexing itself.
