"""Tests for the cheap retrieval metrics and the eval set file. No models needed."""

import json
from pathlib import Path

import pytest

from eval.retrieval_metrics import gold_chunk_sets, gold_ranks, hit_at_k, recall_at_k, reciprocal_rank

CORPUS = {
    "docs/a.md::0": {"source": "docs/a.md", "text": "Well A. Mud weight: 9.2 ppg."},
    "docs/a.md::1": {"source": "docs/a.md", "text": "Mud weight: 9.2 ppg. Overlap, then HSE notes."},
    "docs/b.md::0": {"source": "docs/b.md", "text": "Well B. Mud weight: 9.2 ppg."},
}


def test_gold_chunks_match_by_file_and_text():
    sets = gold_chunk_sets([{"doc": "a.md", "text": "Mud weight: 9.2 ppg."}], CORPUS)
    assert sets == [{"docs/a.md::0", "docs/a.md::1"}]  # both overlapping chunks, never b.md


def test_hit_recall_and_mrr():
    gold = [{"x"}, {"y"}]                 # a 2-chunk (multi-hop) question
    ranked = ["z", "x", "w", "y"]
    assert hit_at_k(ranked, gold, 1) == 0.0
    assert hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 2) == 0.5   # found x, not yet y
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert reciprocal_rank(ranked, gold) == 0.5  # first gold chunk is at rank 2
    assert reciprocal_rank(["z"], gold) == 0.0
    assert gold_ranks(ranked, gold) == [2, 4]


def test_eval_set_shape():
    questions = json.loads((Path(__file__).resolve().parent.parent / "eval" / "eval_set.json").read_text())
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    repeats = [q for q in questions if q["category"] == "repeat"]
    assert len(repeats) == 1 and repeats[0]["repeat_of"] in ids
    # the ONLY repeated question text is the labelled one
    texts = [q["user_input"] for q in questions]
    assert len(texts) - len(set(texts)) == 1
    for q in questions:
        assert (q["category"] == "not_documented") == (not q["evidence"])
