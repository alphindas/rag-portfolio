"""
ingest/bm25_index.py

Builds a BM25 keyword-search index over chunks (from ingest/chunk.py),
persisted so retrieve/hybrid.py can fuse BM25 ranks with ChromaDB's
vector-search ranks via RRF.

Mirrors embed.py's chunk-id scheme exactly:
    ids = f"{c['source']}::{c['chunk_index']}"
This is what makes RRF fusion in hybrid.py possible later -- a BM25
result and a vector-search result for the same chunk must carry the
same id, or fusion can't tell they're the same thing.

Why this can't just re-run chunk_document() on one file and rebuild:
------------------------------------------------------------------
ChromaDB persists incrementally: index doc1.md today, doc2.md tomorrow,
and the collection (via upsert) ends up containing chunks from both.
BM25Okapi has no equivalent -- it's a plain object fit once from a
fixed in-memory list of tokenized documents. There's no "add one more
document to an existing BM25 index" operation.

So instead of rebuilding from scratch on whatever file you pass this
run, we persist the underlying corpus (ids, texts, metadata) alongside
the fitted index. Each run:
  1. loads the existing corpus (if any) from disk
  2. merges in the new chunks, keyed by id (same id overwrites --
     matches Chroma's upsert behavior when you re-ingest an edited doc)
  3. rebuilds BM25Okapi over the FULL merged corpus
  4. persists the merged corpus + rebuilt index

Rebuilding BM25 from scratch each run is deliberately not optimized
further: it's pure-Python term counting over text you already have in
memory, with no GPU/network cost. At portfolio scale (hundreds to a
few thousand chunks) this is milliseconds, so there's no reason to
add incremental-update complexity that real BM25 systems (Lucene,
OpenSearch) already solve properly.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

BM25_PERSIST_DIR = Path("./bm25_index")
BM25_STORE_PATH = BM25_PERSIST_DIR / "bm25_store.pkl"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Lowercase, split on non-alphanumeric boundaries. No stemming or
    stopword removal:
    - Stopwords ("the", "is", "and") get naturally down-weighted by
      BM25's IDF term anyway -- they appear in nearly every chunk, so
      removing them is a minor optimization, not a correctness fix.
    - Stemming would help recall on technical docs with word-form
      variation (e.g. "drilling" / "drill"), but adds a dependency and
      a step that must be applied identically at index time and query
      time. Left out for now as a documented trade-off, not a hidden
      default.
    """
    return _TOKEN_RE.findall(text.lower())


def _make_chunk_id(chunk: dict) -> str:
    """Must exactly match embed.py's id scheme -- this is the shared key
    that lets hybrid.py fuse BM25 and vector-search results later."""
    return f"{chunk['source']}::{chunk['chunk_index']}"


# ---------------------------------------------------------------------------
# Corpus persistence (ids, texts, metadata -- NOT the fitted BM25 object;
# that gets rebuilt fresh from this on every run, see module docstring)
# ---------------------------------------------------------------------------

def _load_corpus() -> dict[str, dict]:
    """
    Returns {chunk_id: {"text": ..., "source": ..., "chunk_index": ...,
    "header_path": ..., "token_count": ...}}. Empty dict if no store
    exists yet (first run).
    """
    if not BM25_STORE_PATH.exists():
        return {}
    with BM25_STORE_PATH.open("rb") as f:
        store = pickle.load(f)
    return store["corpus"]


def _save_corpus(corpus: dict[str, dict]) -> None:
    BM25_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    with BM25_STORE_PATH.open("wb") as f:
        pickle.dump({"corpus": corpus}, f)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_chunks_bm25(chunks: list[dict]) -> None:
    """
    Merge the given chunks (as produced by ingest/chunk.py) into the
    persisted BM25 corpus, keyed by the same id scheme embed.py uses.
    Same id overwrites (matches Chroma's upsert semantics when you
    re-ingest an edited document).
    """
    if not chunks:
        print("[bm25] No chunks to index.")
        return

    corpus = _load_corpus()

    for c in chunks:
        chunk_id = _make_chunk_id(c)
        corpus[chunk_id] = {
            "text": c["text"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "header_path": c["header_path"],
            "token_count": c["token_count"],
        }

    _save_corpus(corpus)
    print(f"[bm25] Merged {len(chunks)} chunks into corpus "
          f"({len(corpus)} total chunks persisted at {BM25_PERSIST_DIR})")


def load_bm25_index() -> dict:
    """
    Loads the persisted corpus and rebuilds BM25Okapi over it fresh.
    Returns a bundle with the fitted index, aligned chunk_ids, and
    metadata -- ready for search_bm25() or for hybrid.py to consume.
    """
    corpus = _load_corpus()
    if not corpus:
        raise FileNotFoundError(
            f"No BM25 corpus found at {BM25_STORE_PATH}. "
            "Run bm25_index.py on at least one document first."
        )

    chunk_ids = list(corpus.keys())
    tokenized_corpus = [tokenize(corpus[cid]["text"]) for cid in chunk_ids]
    metadata = [
        {
            "source": corpus[cid]["source"],
            "header_path": corpus[cid]["header_path"],
        }
        for cid in chunk_ids
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return {
        "bm25": bm25,
        "chunk_ids": chunk_ids,
        "metadata": metadata,
    }

def get_corpus() -> dict[str, dict]:
    """
    Public accessor for the full persisted corpus (text + metadata,
    keyed by chunk_id). Used by retrieve/hybrid.py to enrich fused
    results with actual text -- separate from load_bm25_index(), which
    only returns the stripped-down metadata needed for BM25 scoring.
    """
    return _load_corpus()

def search_bm25(query: str, index_bundle: dict, top_k: int = 10) -> list[tuple[str, float]]:
    """
    Returns [(chunk_id, score), ...] sorted by score descending.
    This is what retrieve/hybrid.py will call before RRF-fusing with
    ChromaDB's vector search results.
    """
    query_tokens = tokenize(query)
    scores = index_bundle["bm25"].get_scores(query_tokens)
    ranked = sorted(
        zip(index_bundle["chunk_ids"], scores),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_k]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from chunk import chunk_document  # local import: ingest/chunk.py

    if len(sys.argv) != 2:
        print("Usage: python bm25_index.py <path-to-markdown-file>")
        sys.exit(1)

    doc_chunks = chunk_document(sys.argv[1])
    index_chunks_bm25(doc_chunks)

    index_bundle = load_bm25_index()
    print(f"[bm25] Index now covers {len(index_bundle['chunk_ids'])} chunks total.")

    print("\nSanity check with a sample query:")
    sample_query = doc_chunks[0]["text"][:50]
    for chunk_id, score in search_bm25(sample_query, index_bundle, top_k=3):
        print(f"  {chunk_id}: {score:.3f}")