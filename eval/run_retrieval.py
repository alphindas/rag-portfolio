# eval/run_retrieval.py
"""
Step 1 of the eval: runs all four retrieval setups for every question and
saves the ranked chunks to eval/retrieved.json, then exits.

    (a) bm25           BM25 keyword search only
    (b) vector         BGE embeddings only
    (c) hybrid         BM25 + vector fused with RRF
    (d) hybrid_rerank  (c), then the cross-encoder reranker re-orders the top 10

Why a separate process from run_eval.py? A live process keeps GPU memory
even after models are "freed". Retrieval loads the embedder + reranker
(~1.5 GB), this process exits, and generation then runs with the whole 4 GB
card free for the LLM.

Run from the project root, after `python ingest/build_index.py`:
    python -m eval.run_retrieval
"""
import json

from eval.retrieval_metrics import gold_chunk_sets
from retrieve.hybrid import hybrid_search
from retrieve.rerank import rerank

import bm25_index  # on sys.path thanks to retrieve.hybrid

RETRIEVE_K = 10  # chunks kept per setup; hit rate / MRR are computed on these


def main():
    with open("eval/eval_set.json", "r") as f:
        questions = json.load(f)
    corpus = bm25_index.get_corpus()

    results = []
    for i, q in enumerate(questions, start=1):
        text = q["user_input"]
        print(f"[{i}/{len(questions)}] retrieving: {text}")
        hybrid = hybrid_search(text, top_k=RETRIEVE_K, mode="hybrid")
        rankings = {
            "bm25": hybrid_search(text, top_k=RETRIEVE_K, mode="bm25"),
            "vector": hybrid_search(text, top_k=RETRIEVE_K, mode="vector"),
            "hybrid": hybrid,
            # rerank() adds scores to the dicts it gets, so give it copies
            "hybrid_rerank": rerank(text, [dict(c) for c in hybrid], top_k=RETRIEVE_K),
        }
        gold = gold_chunk_sets(q["evidence"], corpus)
        if any(not g for g in gold):  # evidence phrase not found in any chunk
            raise SystemExit(f"{q['id']}: gold evidence not found in the index. "
                             "Rebuild it: python ingest/build_index.py")
        results.append({**q, "gold_chunk_ids": [sorted(g) for g in gold], "rankings": rankings})

    with open("eval/retrieved.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved retrieval results for {len(results)} questions to eval/retrieved.json")


if __name__ == "__main__":
    main()
