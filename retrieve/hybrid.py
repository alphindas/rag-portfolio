"""
retrieve/hybrid.py

Fuses BM25 (keyword) and ChromaDB (vector) search results into a single
ranked list, using Reciprocal Rank Fusion (RRF).

Why RRF instead of combining raw scores?
------------------------------------------------------
BM25 scores and cosine-similarity scores live on incomparable scales:
BM25 is an unbounded term-frequency score, cosine similarity here is
roughly [0, 1] (embeddings are normalized). Averaging or weighting
these directly means implicitly deciding how "0.83 cosine" compares to
"14.2 BM25" -- a comparison with no principled answer.

RRF sidesteps this: it only looks at each result's RANK (1st, 2nd,
3rd...) within its own list, never its raw score. A chunk ranked #1 by
BM25 contributes 1/(k+1), ranked #2 contributes 1/(k+2), etc. The fused
score is the sum of these contributions across both lists, so a chunk
that ranks well in EITHER list (or both) rises to the top.

Why k=60?
------------------------------------------------------
The constant from the original RRF paper (Cormack et al., 2009), and
the de facto default in production hybrid search (Elasticsearch,
Weaviate, etc.). It softens the gap between rank 1 and rank 2, so a
chunk has to rank well consistently rather than being rewarded purely
for one list's #1 spot. No dataset-specific tuning done here -- this
is the well-established starting point, not a magic number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))

from bm25_index import load_bm25_index, search_bm25, get_corpus  # noqa: E402
from embed import embed_query, get_chroma_collection  # noqa: E402

RRF_K = 60
CANDIDATES_PER_RETRIEVER = 20  # how many each retriever returns before fusion
FUSED_TOP_K = 10               # how many fused results to hand to reranker later


def _vector_search(query: str, top_k: int) -> list[tuple[str, float]]:
    """
    Returns [(chunk_id, similarity), ...] from ChromaDB, sorted by
    similarity descending.
    """
    collection = get_chroma_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    # Chroma returns cosine DISTANCE (1 - similarity) when hnsw:space is
    # "cosine", not similarity itself. Convert back so higher = better,
    # matching how we read BM25 scores.
    ids = results["ids"][0]
    distances = results["distances"][0]
    similarities = [1.0 - d for d in distances]

    return list(zip(ids, similarities))


def _rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """
    Takes multiple ranked lists of (chunk_id, score) -- score is
    ignored, only position within each list matters -- and returns one
    list of (chunk_id, fused_score), sorted descending.
    """
    fused_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list, start=1):
            fused_scores.setdefault(chunk_id, 0.0)
            fused_scores[chunk_id] += 1.0 / (k + rank)

    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(query: str, top_k: int = FUSED_TOP_K) -> list[dict]:
    """
    Runs BM25 + vector search independently, fuses via RRF, and returns
    the top_k fused results enriched with text + metadata. Enrichment
    pulls from the BM25 corpus (already keyed by chunk_id with text,
    source, header_path) rather than round-tripping to Chroma again.
    """
    bm25_bundle = load_bm25_index()
    bm25_results = search_bm25(query, bm25_bundle, top_k=CANDIDATES_PER_RETRIEVER)
    vector_results = _vector_search(query, top_k=CANDIDATES_PER_RETRIEVER)

    fused = _rrf_fuse([bm25_results, vector_results])[:top_k]

    corpus = get_corpus()  # {chunk_id: {"text", "source", "header_path", ...}}

    enriched = []
    for chunk_id, fused_score in fused:
        chunk_data = corpus.get(chunk_id, {})
        enriched.append(
            {
                "chunk_id": chunk_id,
                "fused_score": fused_score,
                "text": chunk_data.get("text", ""),
                "source": chunk_data.get("source", ""),
                "header_path": chunk_data.get("header_path", ""),
            }
        )
    return enriched


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hybrid.py '<query text>'")
        sys.exit(1)

    query = sys.argv[1]
    results = hybrid_search(query)

    print(f"Top {len(results)} fused results for: {query!r}\n")
    for r in results:
        print(f"[{r['fused_score']:.4f}] {r['header_path']} ({r['chunk_id']})")
        print(f"  {r['text'][:150].replace(chr(10), ' ')}...")
        print()