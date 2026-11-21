# Refuse when the corpus does not answer

## Sub-features
- coarse score screen at `RAG_MIN_SCORE` (default 0.50), refuses without calling the model
- model sentinel `NO_CONTEXT`, refuses when the retrieved material is insufficient
- `refused: true` responses carry `sources: []`

## How to get to it (user POV)
Ask something the corpus has no answer for: 火星上的重力加速度是多少 (low score),
or 如何给 PDF 加密 (high score, wrong material — the hard case).

## Driving it with scripts/verify.py
`ask refused=True (火星上的重力加速度是多少)` proves the score screen.
`ask refused=True (如何给 PDF 加密)` proves the sentinel and is the regression that
motivated it: that question scores 0.523, above an answerable 0.483.

## Gotchas
- Empty index also reports `refused: true`. Distinguish them by `top_score == 0.0`
  and by running `/reindex` first — otherwise a broken index reads as a working
  refuser.
- A refusal with `sources` non-empty violates the contract; check the JSON, not
  the rendered page.
- Raising `RAG_MIN_SCORE` to "improve" refusal will start refusing the English
  question first. Re-measure both groups before touching it.
