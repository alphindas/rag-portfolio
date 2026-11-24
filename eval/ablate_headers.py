# eval/ablate_headers.py
"""
Splits the gain from the two retrieval changes: context headers on chunks
and BM25-weighted RRF. A 2x2 table: {headers off, on} x {plain, weighted}.

"Headers on" numbers come from eval/retrieved.json (the real index).
"Headers off" needs an index without headers, so this script builds a
throwaway one in a temp folder (your real chroma_db/ and bm25_index/ are not
touched), runs the same retrieval on it, and deletes it.

    python -m eval.run_retrieval      # first
    python -m eval.ablate_headers     # GPU: embedder + reranker, no LLM

Writes eval/ablation_factorial.csv and prints a markdown table.
"""
import json
import tempfile
from pathlib import Path

import pandas as pd

from retrieve.hybrid import hybrid_search  # first: it puts ingest/ on sys.path
from retrieve.rerank import rerank
import bm25_index
import embed
from chunk import chunk_document
from eval.ablation import markdown_table, mean, retrieval_scores
from eval.retrieval_metrics import gold_chunk_sets

DOCS = sorted(Path("sample_docs").glob("*.md"))
SETUPS = {"hybrid": "Hybrid", "hybrid_weighted": "Hybrid, BM25-weighted",
          "hybrid_rerank": "Hybrid + reranker", "hybrid_weighted_rerank": "Hybrid, BM25-weighted + reranker"}


def rankings_without_headers(questions: list[dict]) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        # point both indexes at the temp folder (module constants are read at call time)
        embed.CHROMA_PERSIST_DIR = str(Path(tmp) / "chroma")
        bm25_index.BM25_PERSIST_DIR = Path(tmp) / "bm25"
        bm25_index.BM25_STORE_PATH = Path(tmp) / "bm25" / "store.pkl"

        chunks = [c for f in DOCS for c in chunk_document(str(f))]
        for c in chunks:
            c["text"] = c["text"].split("\n", 1)[1]  # drop the "[well | day | section]" line
        embed.index_chunks(chunks)
        bm25_index.index_chunks_bm25(chunks)
        corpus = bm25_index.get_corpus()

        out = []
        for q in questions:
            plain = hybrid_search(q["user_input"], top_k=10, mode="hybrid")
            weighted = hybrid_search(q["user_input"], top_k=10, mode="hybrid_weighted")
            out.append({**q, "gold_chunk_ids": [sorted(g) for g in gold_chunk_sets(q["evidence"], corpus)],
                        "rankings": {"hybrid": plain, "hybrid_weighted": weighted,
                                     "hybrid_rerank": rerank(q["user_input"], [dict(c) for c in plain], top_k=10),
                                     "hybrid_weighted_rerank": rerank(q["user_input"], [dict(c) for c in weighted], top_k=10)}})
        return out


def main():
    with_headers = [q for q in json.load(open("eval/retrieved.json")) if q["evidence"]]
    without = rankings_without_headers(with_headers)

    rows = []
    for setup, label in SETUPS.items():
        weighted = "weighted" in setup
        for headers, qs in (("off", without), ("on", with_headers)):
            s2 = [retrieval_scores(q, setup, 2) for q in qs]
            s3 = [retrieval_scores(q, setup, 3) for q in qs]
            kw = [i for i, q in enumerate(qs) if q["category"] in ("keyword", "keyword_holdout")]
            rows.append({"setup": label,
                         "headers": headers, "weighting": "on" if weighted else "off",
                         "hit_at_2": mean([x["hit"] for x in s2]), "hit_at_3": mean([x["hit"] for x in s3]),
                         "recall_at_3": mean([x["recall"] for x in s3]), "mrr": mean([x["rr"] for x in s3]),
                         "keyword_hit_at_3": mean([s3[i]["hit"] for i in kw])})
    pd.DataFrame(rows).to_csv("eval/ablation_factorial.csv", index=False)

    print(f"\n### Headers x weighting ({len(with_headers)} answerable questions; keyword = all 10 keyword questions)\n")
    for reranked in (False, True):
        part = [r for r in rows if r["setup"].endswith("reranker") == reranked]
        for r in part:
            r["variant"] = f"headers {r['headers']}, weighting {r['weighting']}"
        part.sort(key=lambda r: (r["headers"] == "on", r["weighting"] == "on"))
        print(f"**{'With' if reranked else 'Without'} reranker**\n")
        print(markdown_table(part, {"variant": "Variant", "hit_at_2": "Hit@2", "hit_at_3": "Hit@3",
                                    "recall_at_3": "Recall@3", "mrr": "MRR", "keyword_hit_at_3": "Keyword Hit@3"}))
        print()


if __name__ == "__main__":
    main()
