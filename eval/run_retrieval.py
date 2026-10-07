# eval/run_retrieval.py
"""
Phase 1: runs hybrid_search + rerank for every question in eval_set.json
and writes results to retrieved.json, then exits.

Why a separate process instead of doing this inline in run_eval.py?
In-process cleanup (nulling model refs + gc.collect() + empty_cache())
still left generation OOMing even with ~2.2GB reported free -- a live
process's CUDA context and allocator fragmentation don't fully release
until the process actually exits. So retrieval runs to completion here,
this process exits, and generation runs afterward in a clean process
with the full card available.

Run from the project root:
    python3 -m eval.run_retrieval
"""
import json

from retrieve.hybrid import hybrid_search
from retrieve.rerank import rerank


def main():
    with open("eval/eval_set.json", "r") as f:
        questions = json.load(f)

    results = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] retrieving: {q['user_input']}")
        top10 = hybrid_search(q["user_input"])
        top5 = rerank(q["user_input"], top10)
        results.append({
            "user_input": q["user_input"],
            "reference": q["reference"],
            "retrieved_chunks": top5,
        })

    with open("eval/retrieved.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved retrieval results for {len(results)} questions to eval/retrieved.json")


if __name__ == "__main__":
    main()
