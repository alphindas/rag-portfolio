"""
ingest/build_index.py

One command to (re)build BOTH indexes from every .md file in a folder:

    python ingest/build_index.py                 # uses sample_docs/
    python ingest/build_index.py path/to/docs

It deletes the old Chroma and BM25 indexes first, so stale chunks from
deleted or renamed files can never linger and break chunk-ID matching.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bm25_index import BM25_PERSIST_DIR, index_chunks_bm25  # noqa: E402
from chunk import chunk_document  # noqa: E402
from embed import CHROMA_PERSIST_DIR, index_chunks  # noqa: E402

DEFAULT_DOCS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def build_all(docs_dir: Path) -> None:
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        sys.exit(f"No .md files found in {docs_dir}")

    chunks = []
    for f in files:
        chunks.extend(chunk_document(str(f)))
    print(f"[build] {len(files)} files -> {len(chunks)} chunks")

    shutil.rmtree(CHROMA_PERSIST_DIR, ignore_errors=True)
    shutil.rmtree(BM25_PERSIST_DIR, ignore_errors=True)

    index_chunks(chunks)       # vector index (Chroma)
    index_chunks_bm25(chunks)  # keyword index (BM25)


if __name__ == "__main__":
    build_all(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS_DIR)
