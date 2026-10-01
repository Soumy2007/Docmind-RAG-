"""Step 4 - persistent vector database (ChromaDB, cosine similarity)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .models import Chunk, RetrievedChunk


class VectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.version = 0  # bumped on every change so the BM25 index knows when to rebuild

    # ---- write -----------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self._col.upsert(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=vectors[start : start + batch].tolist(),
                metadatas=[
                    {"source": c.source, "page": c.page, "doc_id": c.doc_id, "index": c.index}
                    for c in part
                ],
            )
        self.version += 1

    def delete_document(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})
        self.version += 1

    # ---- read ------------------------------------------------------------
    def count(self) -> int:
        return self._col.count()

    def has_document(self, doc_id: str) -> bool:
        return bool(self._col.get(where={"doc_id": doc_id}, limit=1)["ids"])

    def list_documents(self) -> list[dict]:
        got = self._col.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for meta in got["metadatas"] or []:
            d = docs.setdefault(
                meta["doc_id"], {"doc_id": meta["doc_id"], "source": meta["source"], "chunks": 0, "pages": set()}
            )
            d["chunks"] += 1
            d["pages"].add(meta["page"])
        out = []
        for d in docs.values():
            d["pages"] = len(d["pages"])
            out.append(d)
        return sorted(out, key=lambda d: d["source"].lower())

    def all_chunks(self) -> list[RetrievedChunk]:
        got = self._col.get(include=["documents", "metadatas"])
        return [
            self._to_chunk(i, doc, meta)
            for i, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        ]

    def query(self, vector: np.ndarray, k: int, doc_ids: Optional[Iterable[str]] = None) -> list[RetrievedChunk]:
        n = min(k, self.count())
        if n == 0:
            return []
        where = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
        res = self._col.query(
            query_embeddings=[vector.tolist()],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            self._to_chunk(i, doc, meta, dense=max(0.0, 1.0 - float(dist)))
            for i, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    @staticmethod
    def _to_chunk(cid: str, doc: str, meta: dict, dense: float = 0.0) -> RetrievedChunk:
        return RetrievedChunk(
            id=cid, text=doc, source=meta["source"], page=int(meta["page"]), doc_id=meta["doc_id"], dense=dense
        )
