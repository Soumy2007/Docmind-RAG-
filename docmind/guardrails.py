"""Trust layer: confidence scoring (before answering) and groundedness check (after answering)."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .textutils import split_sentences, tokenize

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def chunk_confidence(dense: float, coverage: float) -> float:
    """Blend meaning-similarity (60%) with keyword coverage (40%) into a 0..1 score.

    Blending catches two failure modes: a chunk that "sounds similar" but shares no
    real terms, and a chunk that repeats a keyword but is about something else.
    """
    return 0.6 * min(max(dense, 0.0), 1.0) + 0.4 * min(max(coverage, 0.0), 1.0)


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def cited_numbers(text: str) -> list[int]:
    found: set[int] = set()
    for group in CITATION_RE.findall(text):
        found.update(int(n) for n in re.split(r"\s*,\s*", group))
    return sorted(found)


def grounded_label(score: Optional[float]) -> str:
    if score is None:
        return "n/a"
    if score >= 0.65:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def groundedness(answer: str, chunks, embedder) -> tuple[Optional[float], list[str]]:
    """How well is each answer sentence supported by the retrieved context?

    For every sentence we take the max of (semantic similarity to any chunk) and
    (share of its words found in the context), averaged 50/50. Returns the mean
    over sentences and the list of weakly-supported sentences.
    """
    if not chunks:
        return None, []
    body = strip_citations(answer)
    sentences: list[str] = []
    for line in body.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        sentences.extend(s for s in split_sentences(line) if len(tokenize(s)) >= 3)
    if not sentences:
        return None, []

    context_tokens = set(tokenize(" ".join(c.text for c in chunks)))
    chunk_vecs = embedder.encode([c.text for c in chunks])
    sent_vecs = embedder.encode(sentences)
    dense = (sent_vecs @ chunk_vecs.T).max(axis=1)

    supports, weak = [], []
    for sentence, d in zip(sentences, np.clip(dense, 0, 1)):
        toks = tokenize(sentence)
        lexical = sum(t in context_tokens for t in toks) / len(toks)
        support = 0.5 * float(d) + 0.5 * lexical
        supports.append(support)
        if support < 0.40:
            weak.append(sentence)
    return float(np.mean(supports)), weak
