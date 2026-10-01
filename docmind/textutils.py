"""Small text helpers used by chunking, retrieval, guardrails and the UI."""
from __future__ import annotations

import html
import re

STOPWORDS = frozenset(
    """a an the and or but if then of to in on at by for with from as is are was were be been being am
    do does did has have had it its this that these those i you he she we they them their our your my me
    what which who whom whose when where why how can could should would will shall may might must not no
    yes than so such about into over under also any all each per up out there here get got""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_HTML_ENTITY_WORDS = {"amp", "lt", "gt", "quot"}


def _stem(word: str) -> str:
    """Very light stemming so 'days' matches 'day' and 'policies' matches 'policy'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting paragraph breaks."""
    sentences: list[str] = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        sentences.extend(s.strip() for s in _SENT_RE.split(para) if s.strip())
    return sentences


def term_coverage(query: str, text: str) -> float:
    """Fraction of the query's meaningful words that also appear in `text` (0..1)."""
    q_terms = set(tokenize(query))
    if not q_terms:
        return 0.0
    return len(q_terms & set(tokenize(text))) / len(q_terms)


def highlight_html(text: str, query: str) -> str:
    """HTML-escape `text` and wrap words that start with a query term in <mark>."""
    safe = html.escape(text)
    terms = {
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if t not in STOPWORDS and len(t) > 2 and t not in _HTML_ENTITY_WORDS
    }
    if not terms:
        return safe
    ordered = sorted(terms, key=len, reverse=True)
    pattern = re.compile(r"\b((?:" + "|".join(re.escape(t) for t in ordered) + r")\w*)", re.I)
    return pattern.sub(r"<mark>\1</mark>", safe)
