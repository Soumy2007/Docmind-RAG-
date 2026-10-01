"""Step 2 - split pages into sentence-aware, overlapping chunks.

Unlike a naive character splitter, chunks never cut a sentence in half, and every
chunk keeps its page number so answers can cite exact pages.
"""
from __future__ import annotations

from .models import Chunk, Page
from .textutils import split_sentences

MIN_CHUNK_CHARS = 30


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        # a single very long "sentence" (tables, code) is cut hard at `size`
        pieces = [sentence[i : i + size] for i in range(0, len(sentence), size)] if len(sentence) > size else [sentence]
        for piece in pieces:
            if current and current_len + len(piece) + 1 > size:
                chunks.append(" ".join(current))
                # carry the last few sentences into the next chunk (the overlap)
                keep: list[str] = []
                kept_len = 0
                for prev in reversed(current):
                    if kept_len + len(prev) + 1 > overlap:
                        break
                    keep.insert(0, prev)
                    kept_len += len(prev) + 1
                current, current_len = keep, kept_len
            current.append(piece)
            current_len += len(piece) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[Page], size: int = 800, overlap: int = 150) -> list[Chunk]:
    if size < 100:
        raise ValueError("chunk size must be at least 100 characters")
    overlap = max(0, min(overlap, size // 2))

    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        for n, text in enumerate(_pack(split_sentences(page.text), size, overlap)):
            if len(text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    id=f"{page.doc_id}-p{page.page}-c{n}",
                    text=text,
                    source=page.source,
                    page=page.page,
                    doc_id=page.doc_id,
                    index=counter,
                )
            )
            counter += 1
    return chunks
