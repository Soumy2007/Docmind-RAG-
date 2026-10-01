"""Conversation memory: turn follow-up questions into standalone questions."""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .prompts import build_rewrite_messages

logger = logging.getLogger(__name__)

FOLLOW_UP_WORDS = {"it", "its", "they", "them", "their", "this", "that", "those", "these", "he", "she", "his", "her", "same", "above"}
FOLLOW_UP_STARTS = ("and ", "what about", "how about", "also ", "but ", "then ")


def looks_like_follow_up(question: str) -> bool:
    q = question.lower().strip()
    if q.startswith(FOLLOW_UP_STARTS):
        return True
    words = {w.strip("?,.!;:") for w in q.split()}
    return bool(words & FOLLOW_UP_WORDS)


def rewrite_question(
    question: str,
    history: list[dict],
    llm_fn: Optional[Callable[[list[dict]], str]] = None,
) -> str:
    """Return a standalone version of `question`.

    With an LLM: ask it to rewrite. Without one (offline mode): if the question looks like a
    follow-up, prepend the previous user question as context.
    """
    if not history or not looks_like_follow_up(question):
        return question
    if llm_fn is not None:
        try:
            rewritten = llm_fn(build_rewrite_messages(question, history)).strip().strip('"')
            if 3 <= len(rewritten) <= 400:
                return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("Question rewrite failed (%s); using heuristic.", exc)
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return f"{previous} {question}".strip()
