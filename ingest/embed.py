"""
ingest/embed.py

Embeds chunks (from ingest/chunk.py) using BAAI/bge-base-en and persists
them into a local ChromaDB collection for vector search.

Two BGE-specific details that matter and are easy to get wrong:

1. Asymmetric encoding. BGE models are trained with an instruction
   prefix applied ONLY to queries at search time -- documents/passages
   are embedded plain, with no prefix. If you accidentally add the
   prefix to documents too (or forget it on queries), retrieval quality
   degrades silently -- there's no error, just worse rankings. So this
   module exposes two separate functions, embed_documents() and
   embed_query(), rather than one generic embed() that's easy to misuse.

2. VRAM budget. bge-base-en is ~110M params -- fp16 fits comfortably in
   4GB, but batch size still matters if the GPU is shared with anything
   else (a desktop environment, another process). BATCH_SIZE is kept
   modest and adjustable rather than left at a library default that
   assumes a beefier card.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en"

# BGE's documented query instruction. Applied to queries only -- never
# to documents. This exact string is what the model was trained with;
# changing it changes retrieval behavior.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Conservative for a 4GB card. If you hit CUDA OOM, lower this before
# anything else -- it's the single biggest lever on VRAM usage here.
BATCH_SIZE = 16

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "docs"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model once per process, on GPU if available."""
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        if device == "cuda":
            # fp16 roughly halves VRAM and speeds up inference on
            # consumer GPUs, with negligible quality loss for embedding
            # (unlike generation, where precision loss compounds more).
            _model = _model.half()
        print(f"[embed] Loaded {EMBEDDING_MODEL_NAME} on {device}")
    return _model


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document/passage texts. NO instruction prefix --
    BGE documents are embedded plain.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=len(texts) > BATCH_SIZE,
        normalize_embeddings=True,  # cosine similarity assumes unit vectors
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """
    Embed a single search query. Applies BGE's query instruction prefix
    -- this is what makes query/document embeddings comparable in the
    way BGE was trained for. Forgetting this is the most common BGE
    integration bug.
    """
    model = _get_model()
    prefixed = QUERY_INSTRUCTION + text
    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding[0].tolist()


# ---------------------------------------------------------------------------
# ChromaDB persistence
# ---------------------------------------------------------------------------

def get_chroma_collection():
    """
    Returns a persistent ChromaDB collection. Using PersistentClient (not
    the in-memory default) so the index survives across script runs --
    you don't want to re-embed everything just to test a retrieval
    change.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # cosine space to match normalize_embeddings=True above -- Chroma
    # defaults to l2 distance, which is NOT what we want for normalized
    # embeddings, so this is set explicitly rather than left implicit.
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(chunks: list[dict]) -> None:
    """
    Embed a list of chunk dicts (as produced by ingest/chunk.py) and
    upsert them into the ChromaDB collection.

    Chunk IDs are deterministic (source path + chunk_index) so re-running
    ingestion on an updated document overwrites its old chunks instead of
    duplicating them.
    """
    if not chunks:
        print("[embed] No chunks to index.")
        return

    collection = get_chroma_collection()

    texts = [c["text"] for c in chunks]
    ids = [f"{c['source']}::{c['chunk_index']}" for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "header_path": c["header_path"],
            "token_count": c["token_count"],
        }
        for c in chunks
    ]

    print(f"[embed] Embedding {len(texts)} chunks...")
    embeddings = embed_documents(texts)

    # upsert (not add): re-running on the same source file overwrites
    # its existing chunks rather than erroring on duplicate IDs.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[embed] Indexed {len(texts)} chunks into '{CHROMA_COLLECTION_NAME}' "
          f"(persisted at {CHROMA_PERSIST_DIR})")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from chunk import chunk_document  # local import: ingest/chunk.py

    if len(sys.argv) != 2:
        print("Usage: python embed.py <path-to-markdown-file>")
        sys.exit(1)

    doc_chunks = chunk_document(sys.argv[1])
    index_chunks(doc_chunks)

    # Quick sanity check: embed a query and show the collection now has
    # the expected count.
    collection = get_chroma_collection()
    print(f"[embed] Collection now contains {collection.count()} chunks total.")