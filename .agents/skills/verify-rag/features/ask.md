# Ask a question, get a cited answer

## Sub-features
- free-text question box at `GET /`
- `POST /ask` returning `answer`, `sources[]`, `top_score`, `refused`

## How to get to it (user POV)
Open the page, type a question the docs answer, press 提问. Or POST the same JSON.

## Driving it with scripts/verify.py
`drive: POST /ask 200 (退款多久到账)` and the two assertions after it:
`ask refused=False`, `source path contains '退款'`, plus `cited [n] within range`.
Expected shape: `sources` non-empty, every `path` ending in `退款政策.md`,
`top_score` around 0.72.

## Gotchas
- The answer may cite `[3]` while the top hit is `[1]`. That is correct behaviour
  (the model picked the chunk that answers, not the highest-scoring one) — assert
  `1 <= n <= len(sources)`, never `n == 1`.
- A cross-lingual English question returns `top_score` near 0.48, far lower than
  Chinese ones. Do not read a low score as "retrieval broken" without checking
  that the right document still ranks in the top 6 (`RAG_TOP_K`).
