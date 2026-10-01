"""Plain data containers shared by all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Page:
    text: str
    page: int
    source: str
    doc_id: str


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int
    doc_id: str
    index: int


@dataclass
class RetrievedChunk:
    id: str
    text: str
    source: str
    page: int
    doc_id: str
    dense: float = 0.0          # cosine similarity to the question (0..1)
    bm25: float = 0.0           # keyword score
    fused: float = 0.0          # Reciprocal Rank Fusion score
    rerank: Optional[float] = None  # cross-encoder score (if re-ranking is on)
    coverage: float = 0.0       # share of question terms found in this chunk
    confidence: float = 0.0     # blended 0..1 relevance estimate

    @property
    def label(self) -> str:
        return f"{self.source} - page {self.page}"


@dataclass
class Answer:
    answer: str
    refused: bool = False
    confidence: float = 0.0
    groundedness: Optional[float] = None
    grounded_label: str = "n/a"
    unsupported: list[str] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)   # 1-based source numbers the answer cites
    standalone_question: str = ""
    provider: str = "extractive"
    model: str = ""
    latency_ms: int = 0
    note: str = ""
