# Rebuild the index

## Sub-features
- `POST /reindex` over `docs/**/*.md`, returns `documents`, `chunks`, `seconds`, `paths`
- `Store.reset()` + full re-embed; the cached similarity matrix is invalidated

## How to get to it (user POV)
Drop a markdown file into `docs/`, then POST `/reindex`. There is no watcher.

## Driving it with scripts/verify.py
`drive: POST /reindex 200` followed by `drive: index non-empty after reindex`
(`documents=8 chunks=22` on the seed corpus). The second check exists because a
200 with `chunks=0` means every later Ask check is testing an empty index.

## Gotchas
- `documents` counts files, `chunks` counts split pieces; a file with only
  headings contributes 0 chunks and is not an error.
- Two concurrent reindexes share `data/index.db`. Reindex one instance at a time.
- After changing `RAG_EMBED_MODEL`, vectors from the old model remain until
  reindex; search then raises dimension mismatch instead of returning garbage.
