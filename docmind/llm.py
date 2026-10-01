"""Step 6 - the LLM layer.

Every provider is a plain HTTP call, so there are no extra SDKs to install. Swap providers
by changing LLM_PROVIDER in .env or the dropdown in the UI. The special provider
"extractive" needs no model at all: it builds the answer from the best-matching sentences.
"""
from __future__ import annotations

import os

import requests

from .models import RetrievedChunk
from .textutils import split_sentences, tokenize

PROVIDERS = ["extractive", "ollama", "groq", "gemini", "openai", "anthropic"]

# Model names change over time - override with LLM_MODEL or <PROVIDER>_MODEL in .env.
DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
}
KEY_NAMES = {p: f"{p.upper()}_API_KEY" for p in ("groq", "gemini", "openai", "anthropic")}


class LLMError(RuntimeError):
    pass


def resolve_model(provider: str, model: str | None = None) -> str:
    provider = provider.lower()
    return model or os.getenv(f"{provider.upper()}_MODEL") or DEFAULT_MODELS.get(provider, "")


def _key(provider: str) -> str:
    name = KEY_NAMES[provider]
    value = os.getenv(name, "").strip()
    if not value:
        raise LLMError(f"{name} is not set. Add it to your .env file or paste it in the sidebar.")
    return value


def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"Could not reach {url.split('/')[2]}: {exc}") from exc
    if resp.status_code >= 400:
        raise LLMError(f"HTTP {resp.status_code} from {url.split('/')[2]}: {resp.text[:200]}")
    return resp.json()


def chat(messages: list[dict], provider: str, model: str | None = None, temperature: float = 0.1, timeout: int = 90) -> str:
    """Send chat messages [{'role': 'system'|'user'|'assistant', 'content': str}] and return text."""
    provider = provider.lower()
    model = resolve_model(provider, model)

    if provider == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        data = _post(f"{host}/api/chat", {}, {
            "model": model, "messages": messages, "stream": False, "options": {"temperature": temperature},
        }, timeout)
        return data["message"]["content"].strip()

    if provider in ("groq", "openai"):
        url = (
            "https://api.groq.com/openai/v1/chat/completions"
            if provider == "groq"
            else "https://api.openai.com/v1/chat/completions"
        )
        data = _post(url, {"Authorization": f"Bearer {_key(provider)}"},
                     {"model": model, "messages": messages, "temperature": temperature}, timeout)
        return data["choices"][0]["message"]["content"].strip()

    if provider == "gemini":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        payload = {"contents": contents, "generationConfig": {"temperature": temperature}}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": _key("gemini")}, payload, timeout,
        )
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response: {str(data)[:200]}") from exc

    if provider == "anthropic":
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": _key("anthropic"), "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": 1024, "system": system, "messages": turns, "temperature": temperature},
            timeout,
        )
        return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text").strip()

    raise LLMError(f"Unknown provider '{provider}'. Choose from {PROVIDERS}.")


def _units(text: str) -> list[str]:
    """Sentences, but tiny fragments like '1.1 Casual leave.' are merged into the next sentence."""
    units: list[str] = []
    carry = ""
    for sentence in split_sentences(text):
        candidate = f"{carry} {sentence}".strip()
        if len(sentence.split()) < 5:
            carry = candidate
            continue
        units.append(candidate)
        carry = ""
    if carry:
        units.append(carry)
    return units


def extractive_answer(question: str, chunks: list[RetrievedChunk], max_sentences: int = 2) -> str:
    """No-LLM fallback: return the sentences that best match the question, with citations."""
    q_terms = set(tokenize(question))
    scored = []
    for rank, chunk in enumerate(chunks, start=1):
        for pos, unit in enumerate(_units(chunk.text)):
            overlap = len(q_terms & set(tokenize(unit)))
            if overlap:
                # more overlap wins; ties go to higher-ranked chunks and shorter (more precise) sentences
                scored.append((overlap / max(len(q_terms), 1) + 0.1 / rank - len(unit) / 20000, rank, pos, unit))
    if not scored:
        if not chunks:
            return "I could not find this in the documents."
        return " ".join(_units(chunks[0].text)[:2]) + " [1]"
    best = sorted(scored, key=lambda t: t[0], reverse=True)[:max_sentences]
    best.sort(key=lambda t: (t[1], t[2]))  # read them in document order
    return "Based on the documents:\n" + "\n".join(f"- {s} [{rank}]" for _, rank, _, s in best)
