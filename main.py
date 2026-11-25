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

    if result["warning"]:
        print(f"\n[warning] {result['warning']}")
    print("\n=== Answer ===")
    print(result["answer"] or "(no answer)")
    print("\n=== Sources ===")
    print(format_sources(result["chunks"]))
    status = (result["citations"] or {}).get("status")
    if status == "no_citations":
        print("\n[warning] the answer has no [n] citations, so it is not verified against the sources.")
    elif status == "invalid":
        print(f"\n[warning] the answer cites sources that do not exist: {result['citations']['invalid']}")
    print("\nTiming (ms): " + " | ".join(f"{name} {ms:.0f}" for name, ms in result["timings"].items()))
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
