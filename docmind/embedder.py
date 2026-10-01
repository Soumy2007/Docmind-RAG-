"""Step 3 - turn text into vectors.

Uses a sentence-transformers model (semantic, recommended). If it cannot be loaded
(no internet on first run, not installed, ...) it falls back to a small deterministic
"hashing" embedder so the whole app still runs. The fallback is keyword-ish, not truly
semantic - the UI shows which backend is active.
"""
from __future__ import annotations

import logging
import re
import zlib

import numpy as np

from .textutils import tokenize

logger = logging.getLogger(__name__)
HASH_DIM = 1024


class Embedder:
    def __init__(self, model_name: str, backend: str = "auto"):
        self.model_name = model_name
        self._model = None
        if backend != "hashing":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
            except Exception as exc:  # noqa: BLE001 - any failure -> fallback
                logger.warning(
                    "Could not load embedding model '%s' (%s). "
                    "Falling back to the lightweight hashing embedder.",
                    model_name,
                    exc,
                )
        self.dim = self._model.get_sentence_embedding_dimension() if self._model else HASH_DIM

    @property
    def is_semantic(self) -> bool:
        return self._model is not None

    @property
    def backend_name(self) -> str:
        return f"sentence-transformers ({self.model_name})" if self._model else "hashing fallback (keyword-like)"

    @property
    def tag(self) -> str:
        """Used in the Chroma collection name so different models never share an index."""
        raw = self.model_name if self._model else "hashing"
        return re.sub(r"[^a-zA-Z0-9]+", "_", raw)[:50].strip("_")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            vectors = self._model.encode(
                list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vectors, dtype=np.float32)
        return np.vstack([self._hash_vector(t) for t in texts])

    @staticmethod
    def _hash_vector(text: str) -> np.ndarray:
        vec = np.zeros(HASH_DIM, dtype=np.float32)
        tokens = tokenize(text)
        features = [(tok, 1.0) for tok in tokens] + [(f"{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:])]
        for feature, weight in features:
            h = zlib.crc32(feature.encode("utf-8"))
            vec[h % HASH_DIM] += weight if (h >> 16) & 1 else -weight
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
