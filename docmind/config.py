"""Central configuration. Values come from environment variables / the .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    # --- models -----------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    # "auto"    -> use sentence-transformers, fall back to hashing if it cannot load
    # "hashing" -> force the tiny offline embedder (useful for tests / no internet)
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))
    rerank_model: str = field(
        default_factory=lambda: _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # --- chunking ---------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))

    # --- retrieval --------------------------------------------------------
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    candidate_k: int = field(default_factory=lambda: int(_env("CANDIDATE_K", "12")))
    min_confidence: float = field(default_factory=lambda: float(_env("MIN_CONFIDENCE", "0.22")))

    # --- LLM --------------------------------------------------------------
    # extractive | ollama | groq | gemini | openai | anthropic
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "extractive"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))

    # --- paths ------------------------------------------------------------
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env("CHROMA_DIR", str(ROOT / "storage" / "chroma")))
    )
    upload_dir: Path = field(default_factory=lambda: ROOT / "data" / "uploads")
    sample_dir: Path = field(default_factory=lambda: ROOT / "data" / "sample_docs")
    eval_file: Path = field(default_factory=lambda: ROOT / "eval" / "test_questions.json")
