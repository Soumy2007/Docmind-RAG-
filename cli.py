"""Terminal interface.

    python cli.py ingest data/sample_docs        # index a file or folder
    python cli.py chat                           # interactive Q&A
    python cli.py ask "How many casual leave days?"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docmind.pipeline import RAGPipeline


def show(ans) -> None:
    print(f"\n{ans.answer}\n")
    conf = f"confidence {ans.confidence:.0%}"
    ground = f" | groundedness {ans.grounded_label}" if ans.groundedness is not None else ""
    print(f"[{conf}{ground} | {ans.latency_ms} ms | {ans.provider}]")
    if ans.note:
        print(f"note: {ans.note}")
    for i, s in enumerate(ans.sources, 1):
        mark = "*" if i in ans.cited else " "
        print(f" {mark}[{i}] {s.label}  (confidence {s.confidence:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="DocMind RAG command line")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="index a file or a folder")
    ing.add_argument("path")
    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--provider")
    sub.add_parser("chat", help="interactive chat")
    args = parser.parse_args()

    pipe = RAGPipeline()
    if args.cmd == "ingest":
        target = Path(args.path)
        reports = pipe.ingest_folder(target) if target.is_dir() else [pipe.ingest_file(target)]
        for r in reports:
            print(f"{r['file']}: {r['status']} ({r['pages']} pages, {r['chunks']} chunks)")
    elif args.cmd == "ask":
        show(pipe.ask(args.question, provider=args.provider))
    else:
        print(f"Embedding backend: {pipe.embedder.backend_name}\nType a question (or 'exit').")
        history: list[dict] = []
        while True:
            q = input("\nYou: ").strip()
            if q.lower() in {"exit", "quit", ""}:
                break
            ans = pipe.ask(q, history=history)
            show(ans)
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": ans.answer}]


if __name__ == "__main__":
    main()
