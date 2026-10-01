"""Step 5 - find the best chunks for a question.

Pipeline:  dense (meaning) search  +  BM25 (keyword) search
           -> Reciprocal Rank Fusion  -> optional cross-encoder re-ranking
           -> per-chunk confidence score.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import Embedder
from .guardrails import chunk_confidence
from .models import RetrievedChunk
from .textutils import tokenize, term_coverage
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)
RRF_K = 60  # standard constant for Reciprocal Rank Fusion


class Reranker:
    """Optional cross-encoder that re-scores (question, chunk) pairs. Loaded lazily."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._failed = False

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Re-ranker '%s' unavailable (%s). Continuing without it.", self.model_name, exc)
            self._failed = True
            return False

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.predict([(query, t) for t in texts])]


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, reranker: Optional[Reranker] = None):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[RetrievedChunk] = []
        self._bm25_key: tuple[int, int] = (-1, -1)

    # ---- BM25 (keyword) index -------------------------------------------
    def _ensure_bm25(self) -> None:
        key = (self.store.version, self.store.count())
        if key == self._bm25_key:
            return
        self._corpus = self.store.all_chunks()
        tokens = [tokenize(c.text) or ["_"] for c in self._corpus]
        self._bm25 = BM25Okapi(tokens) if tokens else None
        self._bm25_key = key

    def _bm25_hits(self, query: str, k: int, doc_ids: Optional[Iterable[str]]) -> list[RetrievedChunk]:
        self._ensure_bm25()
        q_tokens = tokenize(query)
        if self._bm25 is None or not q_tokens:
            return []
        allowed = set(doc_ids) if doc_ids else None
        scores = self._bm25.get_scores(q_tokens)
        hits: list[RetrievedChunk] = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] <= 0:
                break
            chunk = self._corpus[int(idx)]
            if allowed is not None and chunk.doc_id not in allowed:
                continue
            hits.append(
                RetrievedChunk(
                    id=chunk.id, text=chunk.text, source=chunk.source, page=chunk.page,
                    doc_id=chunk.doc_id, bm25=float(scores[idx]),
                )
            )
            if len(hits) >= k:
                break
        return hits

    # ---- main entry ------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        candidate_k: int = 12,
        doc_ids: Optional[Iterable[str]] = None,
        hybrid: bool = True,
        rerank: bool = False,
    ) -> list[RetrievedChunk]:
        if self.store.count() == 0 or not query.strip():
            return []
        doc_ids = list(doc_ids) if doc_ids else None
        candidate_k = max(candidate_k, top_k)

        q_vec = self.embedder.encode([query])[0]
        dense_hits = self.store.query(q_vec, candidate_k, doc_ids)
        pool: dict[str, RetrievedChunk] = {h.id: h for h in dense_hits}
        dense_ids = set(pool)
        rank_lists = [[h.id for h in dense_hits]]

        if hybrid:
            bm25_hits = self._bm25_hits(query, candidate_k, doc_ids)
            for hit in bm25_hits:
                if hit.id in pool:
                    pool[hit.id].bm25 = hit.bm25
                else:
                    pool[hit.id] = hit
            rank_lists.append([h.id for h in bm25_hits])

        # Reciprocal Rank Fusion: reward chunks that rank high in ANY list, most if in BOTH
        fused: dict[str, float] = defaultdict(float)
        for ranking in rank_lists:
            for rank, cid in enumerate(ranking):
                fused[cid] += 1.0 / (RRF_K + rank + 1)
        for cid, score in fused.items():
            pool[cid].fused = score
        ordered = sorted(pool.values(), key=lambda c: c.fused, reverse=True)[:candidate_k]

        # chunks found only by BM25 have no dense similarity yet - compute it
        missing = [c for c in ordered if c.id not in dense_ids]
        if missing:
            sims = self.embedder.encode([c.text for c in missing]) @ q_vec
            for chunk, sim in zip(missing, sims):
                chunk.dense = max(0.0, float(sim))

        if rerank and self.reranker is not None and self.reranker.available:
            for chunk, score in zip(ordered, self.reranker.score(query, [c.text for c in ordered])):
                chunk.rerank = score
            ordered.sort(key=lambda c: c.rerank if c.rerank is not None else -1e9, reverse=True)

        final = ordered[:top_k]
        for chunk in final:
            chunk.coverage = term_coverage(query, chunk.text)
            chunk.confidence = chunk_confidence(chunk.dense, chunk.coverage)
        return final
