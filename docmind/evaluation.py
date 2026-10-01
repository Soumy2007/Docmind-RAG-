"""Measure retrieval quality instead of guessing.

Metrics
  Hit@k              did a chunk containing the gold evidence appear in the top-k?
  MRR                mean reciprocal rank of the first correct chunk (1.0 = always rank 1)
  Refusal accuracy   on unanswerable questions, how often did the guardrail correctly refuse?
  Answered rate      on answerable questions, how often did we (correctly) not refuse?

Run from a terminal:
    python -m docmind.evaluation            # compare retrieval modes
    python -m docmind.evaluation --sweep    # also compare chunk sizes
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

from .config import Settings
from .embedder import Embedder
from .pipeline import RAGPipeline

MODES = {
    "Dense only": {"hybrid": False, "rerank": False},
    "Hybrid (dense + BM25)": {"hybrid": True, "rerank": False},
    "Hybrid + re-rank": {"hybrid": True, "rerank": True},
}


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def evaluate(
    pipeline: RAGPipeline,
    questions: list[dict],
    k: int = 4,
    include_rerank: bool = False,
    min_confidence: Optional[float] = None,
) -> dict:
    threshold = pipeline.settings.min_confidence if min_confidence is None else min_confidence
    answerable = [q for q in questions if q.get("should_answer", True)]
    unanswerable = [q for q in questions if not q.get("should_answer", True)]
    modes = {name: cfg for name, cfg in MODES.items() if include_rerank or not cfg["rerank"]}

    summary, details = [], []
    for mode, cfg in modes.items():
        hits, reciprocal, answered_ok = 0, 0.0, 0
        for q in answerable:
            chunks = pipeline.retrieve(q["question"], top_k=k, **cfg)
            evidence = _norm(q["evidence"])
            rank = next((i for i, c in enumerate(chunks, 1) if evidence in _norm(c.text)), None)
            confident = bool(chunks) and max(c.confidence for c in chunks) >= threshold
            hits += rank is not None
            reciprocal += 1.0 / rank if rank else 0.0
            answered_ok += confident
            details.append({"mode": mode, "question": q["question"], "answerable": True,
                            "found_at_rank": rank, "confidence": round(max((c.confidence for c in chunks), default=0), 2)})

        refused_ok = 0
        for q in unanswerable:
            chunks = pipeline.retrieve(q["question"], top_k=k, **cfg)
            conf = max((c.confidence for c in chunks), default=0.0)
            refused_ok += conf < threshold
            details.append({"mode": mode, "question": q["question"], "answerable": False,
                            "found_at_rank": None, "confidence": round(conf, 2)})

        summary.append({
            "Mode": mode,
            f"Hit@{k}": round(hits / max(len(answerable), 1), 3),
            "MRR": round(reciprocal / max(len(answerable), 1), 3),
            "Answered rate": round(answered_ok / max(len(answerable), 1), 3),
            "Refusal accuracy": round(refused_ok / max(len(unanswerable), 1), 3) if unanswerable else None,
        })
    return {"summary": summary, "details": details}


def sweep_chunk_sizes(sample_dir: Path, questions: list[dict], sizes=(300, 500, 800, 1200), k: int = 4) -> list[dict]:
    """Rebuild a throw-away index for each chunk size and compare hybrid retrieval quality."""
    base = Settings()
    embedder = Embedder(base.embedding_model, base.embedding_backend)  # load once, reuse
    rows = []
    for size in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(chroma_dir=Path(tmp), chunk_size=size, chunk_overlap=size // 5)
            pipe = RAGPipeline(settings, embedder=embedder)
            pipe.ingest_folder(sample_dir)
            n_chunks = pipe.store.count()
            result = evaluate(pipe, questions, k=k)
            hybrid = next(r for r in result["summary"] if r["Mode"].startswith("Hybrid"))
            rows.append({"Chunk size": size, "Chunks": n_chunks, **{key: v for key, v in hybrid.items() if key != "Mode"}})
    return rows


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    widths = {c: max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(str(c).ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocMind retrieval quality")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rerank", action="store_true", help="also test the cross-encoder re-ranker")
    parser.add_argument("--sweep", action="store_true", help="compare several chunk sizes")
    args = parser.parse_args()

    settings = Settings()
    questions = load_questions(settings.eval_file)
    out: dict = {}

    with tempfile.TemporaryDirectory() as tmp:  # isolated index; your real one is untouched
        pipe = RAGPipeline(Settings(chroma_dir=Path(tmp)))
        print(f"Embedding backend: {pipe.embedder.backend_name}")
        pipe.ingest_folder(settings.sample_dir)
        result = evaluate(pipe, questions, k=args.k, include_rerank=args.rerank)
        print("\nRetrieval modes")
        _print_table(result["summary"])
        out["modes"] = result["summary"]

    if args.sweep:
        rows = sweep_chunk_sizes(settings.sample_dir, questions, k=args.k)
        print("\nChunk-size sweep (hybrid retrieval)")
        _print_table(rows)
        out["chunk_sweep"] = rows

    target = settings.eval_file.parent / "results.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {target}")


if __name__ == "__main__":
    main()
