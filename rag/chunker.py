"""Markdown-aware chunking.

One job: turn a document into chunks that each fit the embedding window and
carry enough heading context to be readable on their own. No retrieval logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str
    ord: int


@dataclass(frozen=True)
class Block:
    heading: str
    body: str


def _blocks(text: str) -> list[Block]:
    """Split on headings, keeping the heading path (parent > child)."""
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            blocks.append(Block(heading=heading, body=body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading = " > ".join(t for _, t in stack)
        else:
            buf.append(line)
    flush()
    return blocks


def _windows(body: str, max_chars: int, overlap: int) -> list[str]:
    """Hard-wrap one block, preferring paragraph then sentence boundaries."""
    body = body.strip()
    if len(body) <= max_chars:
        return [body]

    pieces: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", body):
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = ""
        while len(para) > max_chars:
            cut = para.rfind("。", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else para.rfind(". ", 0, max_chars)
            cut = cut + 2 if cut > max_chars // 2 and not para[:cut].endswith("。") else cut
            if cut <= max_chars // 2:
                cut = max_chars
            pieces.append(para[:cut].strip())
            para = para[max(0, cut - overlap) :].strip()
        current = para
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def chunk_markdown(text: str, *, max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    chunks: list[Chunk] = []
    for block in _blocks(text):
        for piece in _windows(block.body, max_chars, overlap):
            chunks.append(Chunk(text=piece, heading=block.heading, ord=len(chunks)))
    return chunks
