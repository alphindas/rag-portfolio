"""
Smoke tests. No GPU, no Ollama, no Groq: the embedding model is replaced
by a fake in the one test that touches Chroma. (Chunking still loads the
BGE *tokenizer*, a small file that is cached after the first download.)
"""

from pathlib import Path

import pytest

import bm25_index
import chunk
import embed
from chunk import chunk_document
from retrieve.hybrid import _rrf_fuse, has_exact_token

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "ddr_condor7_day04.md"


@pytest.fixture
def needs_tokenizer():
    """Chunking needs the BGE tokenizer. If it is not cached and there is
    no internet, skip instead of failing: a missing network is not a bug."""
    try:
        chunk._get_tokenizer()
    except Exception as e:  # offline errors differ between library versions
        pytest.skip(f"BGE tokenizer unavailable (offline?): {type(e).__name__}")


@pytest.mark.usefixtures("needs_tokenizer")
def test_chunking_sample_doc():
    chunks = chunk_document(str(SAMPLE_DOC))
    assert chunks, "expected at least one chunk"
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # source is repo-relative, so IDs are the same on every machine
    assert all(c["source"] == "sample_docs/ddr_condor7_day04.md" for c in chunks)
    assert all(c["text"].strip() for c in chunks)


@pytest.mark.usefixtures("needs_tokenizer")
def test_chunk_ids_match_between_bm25_and_chroma(tmp_path, monkeypatch):
    # Send both indexes to a temp folder and fake the embeddings.
    monkeypatch.setattr(embed, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(bm25_index, "BM25_PERSIST_DIR", tmp_path / "bm25")
    monkeypatch.setattr(bm25_index, "BM25_STORE_PATH", tmp_path / "bm25" / "store.pkl")
    monkeypatch.setattr(embed, "embed_documents", lambda texts: [[1.0, 0.0, 0.0]] * len(texts))

    chunks = chunk_document(str(SAMPLE_DOC))
    embed.index_chunks(chunks)
    bm25_index.index_chunks_bm25(chunks)

    chroma_ids = set(embed.get_chroma_collection().get()["ids"])
    bm25_ids = set(bm25_index.load_bm25_index()["chunk_ids"])
    assert chroma_ids == bm25_ids
    assert len(chroma_ids) == len(chunks)


def test_rrf_fusion_order():
    bm25 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    vector = [("b", 0.9), ("d", 0.8), ("a", 0.7)]
    fused = _rrf_fuse([bm25, vector])
    ids = [chunk_id for chunk_id, _ in fused]

    # a (rank 1 + rank 3) and b (rank 2 + rank 1) appear in BOTH lists,
    # so they beat c and d, which appear in only one. b edges out a.
    assert ids[:2] == ["b", "a"]
    assert set(ids[2:]) == {"c", "d"}
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)  # k=60: b is rank 2 and rank 1


def test_weighted_rrf_lets_bm25_win_on_exact_queries():
    bm25 = [("target", 9.0), ("other", 5.0)]
    vector = [("other", 0.9), ("noise", 0.8)]
    plain = [cid for cid, _ in _rrf_fuse([bm25, vector])]
    weighted = [cid for cid, _ in _rrf_fuse([bm25, vector], weights=[100.0, 1.0])]
    assert plain[0] == "other"      # in both lists, so equal weights prefer it
    assert weighted[0] == "target"  # BM25's #1 wins when BM25 counts for more


def test_exact_token_detection():
    for q in ["work order WO-98389?", "bit serial number BS-85748", "chlorides of 19,200 mg/L", "depth of 10,955 ft"]:
        assert has_exact_token(q), q
    for q in ["mud weight on Condor-9 on Day 4", "Falcon-1 Day 12 near Rig 88", "pressure of 9.2 ppg"]:
        assert not has_exact_token(q), q
