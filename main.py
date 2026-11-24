"""
main.py -- ask questions from the command line.

    python main.py                      # interactive: type questions, Ctrl+C or 'quit' to stop
    python main.py --question "..."     # one question, then exit

Build the indexes first:  python ingest/build_index.py
"""

import argparse

from pipeline import format_sources, run_query


def ask(question: str) -> None:
    try:
        result = run_query(question)
    except FileNotFoundError as e:  # BM25 store missing
        raise SystemExit(f"{e}\nBuild the indexes first: python ingest/build_index.py")
    except ConnectionError:
        raise SystemExit("Could not reach Ollama. Is it running? (ollama serve)")

    print("\n=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    print(format_sources(result["chunks"]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG system a question.")
    parser.add_argument("-q", "--question", help="ask one question and exit")
    args = parser.parse_args()

    if args.question:
        ask(args.question)
        return

    print("Ask a question about the documents (empty line or 'quit' to exit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        ask(question)


if __name__ == "__main__":
    main()
