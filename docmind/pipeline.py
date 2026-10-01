"""The orchestrator that ties every step together."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import llm
from .chunker import chunk_pages
from .config import Settings
from .embedder import Embedder
from .guardrails import cited_numbers, grounded_label, groundedness
from .loader import SUPPORTED_EXTENSIONS, file_hash, load_pages
from .memory import rewrite_question
from .models import Answer, RetrievedChunk
from .prompts import NO_ANSWER, build_qa_messages
from .retriever import HybridRetriever, Reranker
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Optional[Settings] = None, embedder: Optional[Embedder] = None):
        self.settings = settings or Settings()
        self.embedder = embedder or Embedder(self.settings.embedding_model, self.settings.embedding_backend)
        self.store = VectorStore(self.settings.chroma_dir, f"docmind_{self.embedder.tag}")
        self.reranker = Reranker(self.settings.rerank_model)
        self.retriever = HybridRetriever(self.store, self.embedder, self.reranker)

    # ------------------------------------------------------------------ ingest
    def ingest_file(
        self,
        path: str | Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> dict:
        """Read -> clean -> chunk -> embed -> store. Returns a small report dict."""
        path = Path(path)
        report = {"file": path.name, "doc_id": "", "status": "", "pages": 0, "chunks": 0}
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report["status"] = f"unsupported type {path.suffix}"
            return report

        doc_id = file_hash(path)
        report["doc_id"] = doc_id
        if self.store.has_document(doc_id):
            report["status"] = "already indexed"
            return report

        if progress:
            progress(0.05, f"Reading {path.name}")
        pages = load_pages(path, doc_id)
        report["pages"] = len(pages)
        if not pages:
            report["status"] = "no extractable text (scanned PDF? try OCR)"
            return report

        chunks = chunk_pages(
            pages,
            size=chunk_size or self.settings.chunk_size,
            overlap=self.settings.chunk_overlap if chunk_overlap is None else chunk_overlap,
        )
        report["chunks"] = len(chunks)

        batch = 64
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            self.store.add_chunks(part, self.embedder.encode([c.text for c in part]))
            if progress:
                progress(0.1 + 0.9 * min(1.0, (start + batch) / len(chunks)), f"Embedding {path.name}")
        report["status"] = "indexed"
        return report

    def ingest_folder(self, folder: str | Path, **kwargs) -> list[dict]:
        reports = []
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                reports.append(self.ingest_file(p, **kwargs))
        return reports

    def remove_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)

    def list_documents(self) -> list[dict]:
        return self.store.list_documents()

    # ---------------------------------------------------------------- retrieve
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_ids: Optional[Iterable[str]] = None,
        hybrid: bool = True,
        rerank: bool = False,
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            query,
            top_k=top_k or self.settings.top_k,
            candidate_k=self.settings.candidate_k,
            doc_ids=doc_ids,
            hybrid=hybrid,
            rerank=rerank,
        )

    # --------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        history: Optional[list[dict]] = None,
        doc_ids: Optional[Iterable[str]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        hybrid: bool = True,
        rerank: bool = False,
        min_confidence: Optional[float] = None,
    ) -> Answer:
        started = time.perf_counter()
        provider = (provider or self.settings.llm_provider).lower()
        model_name = llm.resolve_model(provider, model or self.settings.llm_model or None) if provider != "extractive" else ""
        threshold = self.settings.min_confidence if min_confidence is None else min_confidence
        answer = Answer(answer="", provider=provider, model=model_name, standalone_question=question)

        def finish(a: Answer) -> Answer:
            a.latency_ms = int((time.perf_counter() - started) * 1000)
            return a

        if self.store.count() == 0:
            answer.answer = "No documents are indexed yet. Upload a PDF in the sidebar first."
            answer.refused = True
            return finish(answer)

        llm_fn = None if provider == "extractive" else (lambda msgs: llm.chat(msgs, provider, model_name))

        # 1) make follow-up questions standalone
        answer.standalone_question = rewrite_question(question, history or [], llm_fn)

        # 2) retrieve (hybrid + optional re-rank)
        chunks = self.retrieve(answer.standalone_question, top_k, doc_ids, hybrid, rerank)
        answer.sources = chunks
        answer.confidence = max((c.confidence for c in chunks), default=0.0)

        # 3) guardrail: refuse when nothing relevant was found
        if not chunks or answer.confidence < threshold:
            answer.answer = NO_ANSWER
            answer.refused = True
            answer.note = (
                f"Confidence {answer.confidence:.0%} is below the {threshold:.0%} threshold, "
                "so no answer was generated. The closest passages are shown below."
            )
            return finish(answer)

        # 4) generate
        if provider == "extractive":
            text = llm.extractive_answer(answer.standalone_question, chunks)
        else:
            try:
                text = llm.chat(build_qa_messages(answer.standalone_question, chunks), provider, model_name)
            except llm.LLMError as exc:
                answer.note = f"LLM call failed ({exc}). Showing an extractive answer instead."
                answer.provider = "extractive"
                text = llm.extractive_answer(answer.standalone_question, chunks)
        answer.answer = text
        answer.refused = NO_ANSWER.lower().rstrip(".") in text.lower()[:200]

        # 5) verify the answer against the sources
        if not answer.refused:
            answer.cited = [n for n in cited_numbers(text) if 1 <= n <= len(chunks)]
            score, weak = groundedness(text, chunks, self.embedder)
            answer.groundedness = score
            answer.grounded_label = grounded_label(score)
            answer.unsupported = weak
        return finish(answer)
