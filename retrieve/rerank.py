"""
retrieve/rerank.py

Re-scores hybrid_search()'s fused candidates using a cross-encoder
(BAAI/bge-reranker-base), and returns them re-sorted by that score.

Why rerank at all, if hybrid_search() already ranked things?
------------------------------------------------------
hybrid_search() ranks using a BI-ENCODER (BGE embeddings) and BM25 --
both score the query and each chunk INDEPENDENTLY, then compare. That's
what makes them fast enough to search a whole corpus, but it also means
the model never actually sees the query and chunk together, so it can
miss relevance signals that only show up from direct comparison (e.g.
"button" appears in both, but the *chunk* is about styling buttons, not
click handlers).

A cross-encoder reranker takes (query, chunk) as ONE combined input and
outputs a single relevance score. Far more accurate -- but far too slow
to run over an entire corpus. So the pattern is:
  1. hybrid_search() cheaply narrows thousands of chunks -> ~10 candidates
  2. rerank() expensively re-scores just those ~10 with the cross-encoder
  3. Keep the top few (e.g. top 3-5) to actually feed the LLM

This two-stage "retrieve cheap, rerank precise" pattern is standard in
production retrieval systems for exactly this cost/accuracy trade-off.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder once per process, on GPU if available.

    CrossEncoder auto-detects CUDA the same way SentenceTransformer does,
    so no explicit device handling needed here -- it mirrors embed.py's
    lazy-singleton pattern for consistency.
    """
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
        print(f"[rerank] Loaded {RERANKER_MODEL_NAME}")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 2) -> list[dict]:
    """
    Takes the list of dicts returned by hybrid_search() (each with at
    least "text"), scores each against the query with the cross-encoder,
    and returns the top_k re-sorted by that score (descending).

    Each returned dict keeps its original fields (chunk_id, source,
    header_path, fused_score, text) plus a new "rerank_score".
    """
    if not candidates:
        return []

    model = _get_reranker()

    # CrossEncoder expects a list of [query, passage] pairs -- one pair
    # per candidate, scored independently but batched for efficiency.
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from hybrid import hybrid_search  # local import: retrieve/hybrid.py

    if len(sys.argv) != 2:
        print("Usage: python rerank.py '<query text>'")
        sys.exit(1)

    query = sys.argv[1]

    fused = hybrid_search(query)
    print(f"[rerank] Got {len(fused)} fused candidates from hybrid_search()\n")

    reranked = rerank(query, fused, top_k=5)

    print(f"Top {len(reranked)} reranked results for: {query!r}\n")
    for r in reranked:
        print(f"[rerank={r['rerank_score']:.4f}  fused={r['fused_score']:.4f}] "
              f"{r['header_path']} ({r['chunk_id']})")
        print(f"  {r['text'][:150].replace(chr(10), ' ')}...")
        print()