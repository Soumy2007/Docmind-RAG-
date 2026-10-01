"""FastAPI backend.   Run:  uvicorn api:app --reload      Docs: http://127.0.0.1:8000/docs"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from docmind.loader import SUPPORTED_EXTENSIONS
from docmind.pipeline import RAGPipeline

app = FastAPI(title="DocMind RAG API", version="1.0.0")
_pipeline: Optional[RAGPipeline] = None


def pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class ChatTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    history: list[ChatTurn] = []
    doc_ids: Optional[list[str]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    top_k: int = Field(default=4, ge=1, le=10)
    hybrid: bool = True
    rerank: bool = False
    min_confidence: Optional[float] = Field(default=None, ge=0, le=1)


@app.get("/health")
def health():
    p = pipeline()
    return {"status": "ok", "embedding_backend": p.embedder.backend_name, "chunks": p.store.count()}


@app.get("/documents")
def documents():
    return pipeline().list_documents()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Use one of {sorted(SUPPORTED_EXTENSIONS)}")
    p = pipeline()
    p.settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = p.settings.upload_dir / name
    with open(target, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return p.ingest_file(target)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    pipeline().remove_document(doc_id)
    return {"deleted": doc_id}


@app.post("/ask")
def ask(req: AskRequest):
    ans = pipeline().ask(
        req.question,
        history=[t.model_dump() for t in req.history],
        doc_ids=req.doc_ids,
        provider=req.provider,
        model=req.model,
        top_k=req.top_k,
        hybrid=req.hybrid,
        rerank=req.rerank,
        min_confidence=req.min_confidence,
    )
    return asdict(ans)
