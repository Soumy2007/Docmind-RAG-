"""Step 1 - read files and turn them into clean, page-tagged text."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import Page

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def file_hash(path: str | Path) -> str:
    """Short content hash. Used as the document id so re-uploading the same file is a no-op."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")              # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)        # join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # single newline inside a paragraph -> space
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_pages(path: Path, doc_id: str) -> list[Page]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24
    except ImportError:  # older versions
        import fitz

    pages: list[Page] = []
    with fitz.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [clean_text(b[4]) for b in blocks if b[6] == 0]  # b[6]==0 -> text block
            text = "\n\n".join(p for p in paragraphs if p)
            if text.strip():
                pages.append(Page(text=text, page=number, source=path.name, doc_id=doc_id))
    return pages


def _text_pages(path: Path, doc_id: str) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    paragraphs = [clean_text(p) for p in raw.split("\n\n")]
    text = "\n\n".join(p for p in paragraphs if p)
    return [Page(text=text, page=1, source=path.name, doc_id=doc_id)] if text else []


def load_pages(path: str | Path, doc_id: str) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf_pages(path, doc_id)
    if ext in {".txt", ".md"}:
        return _text_pages(path, doc_id)
    raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
