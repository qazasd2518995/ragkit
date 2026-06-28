"""Command-line interface for ragkit.

Examples
--------
    # Index documents then ask a one-off question
    python -m ragkit.cli ask "What is the refund policy?" --docs ./samples

    # Interactive chat over a folder of documents
    python -m ragkit.cli chat --docs ./samples
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .engine import RagEngine
from .loaders import iter_documents


def _build_engine(doc_paths: list[str]) -> RagEngine:
    engine = RagEngine()
    docs = iter_documents(doc_paths)
    if not docs:
        print(f"⚠️  No supported documents found in {doc_paths}", file=sys.stderr)
        sys.exit(1)
    total = 0
    for text, source in docs:
        n = engine.add_document(text, source)
        total += n
        print(f"  indexed {n:>4} chunks  ←  {source}", file=sys.stderr)
    print(f"✓ Indexed {total} chunks from {len(docs)} document(s)\n", file=sys.stderr)
    return engine


def _print_answer(ans) -> None:
    print(ans.text)
    if ans.citations:
        print("\nSources:")
        for c in ans.citations:
            print(f"  [{c.marker}] {c.source}  (score {c.score})")
            print(f"      {c.snippet}")


def cmd_ask(args: argparse.Namespace) -> None:
    engine = _build_engine(args.docs)
    ans = engine.ask(args.question)
    _print_answer(ans)


def cmd_chat(args: argparse.Namespace) -> None:
    engine = _build_engine(args.docs)
    history: list[dict] = []
    print("💬 Chat mode — ask questions about your documents (Ctrl-D to exit)\n")
    while True:
        try:
            q = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if not q:
            continue
        ans = engine.ask(q, history=history)
        print(f"\nbot › ", end="")
        _print_answer(ans)
        print()
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans.text})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragkit", description="Agentic RAG over your documents, powered by Groq.")
    parser.add_argument("--version", action="version", version=f"ragkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question")
    p_ask.add_argument("--docs", nargs="+", required=True, help="files or folders to index")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--docs", nargs="+", required=True, help="files or folders to index")
    p_chat.set_defaults(func=cmd_chat)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
