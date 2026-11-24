"""
eval/retrieval_metrics.py

Cheap retrieval metrics. No LLM, no GPU, no API: just "did the right chunk
show up, and how high?" Used by the ablation and covered by tests.

A question's gold evidence is a list of {"doc": file name, "text": exact
phrase}. The gold CHUNKS are whichever indexed chunks of that file contain
the phrase (worked out from the live index, so changing the chunker does
not break the eval set).
"""

from pathlib import Path


def gold_chunk_sets(evidence: list[dict], corpus: dict[str, dict]) -> list[set[str]]:
    """One set of chunk IDs per evidence item (a phrase can sit in 2 chunks
    when chunks overlap)."""
    return [
        {cid for cid, c in corpus.items()
         if Path(c["source"]).name == e["doc"] and e["text"] in c["text"]}
        for e in evidence
    ]


def hit_at_k(ranked_ids: list[str], gold_sets: list[set[str]], k: int) -> float:
    """1.0 if ANY gold chunk is in the top k."""
    gold = set().union(*gold_sets)
    return float(any(cid in gold for cid in ranked_ids[:k]))


def recall_at_k(ranked_ids: list[str], gold_sets: list[set[str]], k: int) -> float:
    """Fraction of evidence items found in the top k. For a 2-chunk
    (multi-hop) question: 0.5 if one of the two was found, 1.0 if both."""
    top = set(ranked_ids[:k])
    return sum(bool(g & top) for g in gold_sets) / len(gold_sets)


def doc_hit_at_k(ranked_sources: list[str], gold_docs: list[str], k: int) -> float:
    """Looser than hit_at_k: right FILE in the top k, any chunk of it."""
    return float(any(Path(s).name in gold_docs for s in ranked_sources[:k]))


def reciprocal_rank(ranked_ids: list[str], gold_sets: list[set[str]]) -> float:
    """1 / rank of the first gold chunk (0 if it is not in the list)."""
    gold = set().union(*gold_sets)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def gold_ranks(ranked_ids: list[str], gold_sets: list[set[str]]) -> list:
    """Best rank of each evidence item, or None if not retrieved. For debugging."""
    return [next((r for r, cid in enumerate(ranked_ids, 1) if cid in g), None) for g in gold_sets]
