"""Settings for the local RAG MVP.

Everything is overridable by environment variable; a `.env` file in the project
root is read if present, so nothing here needs a config library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # generation: the agent2api gateway (OpenAI compatible)
    base_url: str = os.environ.get("RAG_BASE_URL", "http://192.168.2.4:3065/v1")
    api_key: str = os.environ.get("RAG_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    model: str = os.environ.get("RAG_MODEL", "deepseek-v4.1-flash")

    # retrieval: local embeddings, no key, no network after the first download
    embed_model: str = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

    docs_dir: Path = Path(os.environ.get("RAG_DOCS_DIR", str(ROOT / "docs")))
    db_path: Path = Path(os.environ.get("RAG_DB_PATH", str(ROOT / "data" / "index.db")))

    top_k: int = int(os.environ.get("RAG_TOP_K", "4"))
    min_score: float = float(os.environ.get("RAG_MIN_SCORE", "0.40"))
    max_chars: int = int(os.environ.get("RAG_MAX_CHARS", "600"))
    overlap: int = int(os.environ.get("RAG_OVERLAP", "80"))
    timeout: float = float(os.environ.get("RAG_TIMEOUT", "90"))


settings = Settings()
