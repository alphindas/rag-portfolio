"""The synthetic corpus must be reproducible: same seed -> same documents."""

import random

import generate_docs as gd


def test_same_seed_gives_same_reports():
    first = gd.build_reports(random.Random(gd.SEED))
    second = gd.build_reports(random.Random(gd.SEED))
    assert {k: r["text"] for k, r in first.items()} == {k: r["text"] for k, r in second.items()}
    assert len(first) == 36


def test_different_seed_gives_different_reports():
    a = gd.build_reports(random.Random(1))
    b = gd.build_reports(random.Random(2))
    assert any(a[k]["text"] != b[k]["text"] for k in a)


def test_committed_files_match_the_generator():
    """If this fails, someone edited sample_docs/ by hand or changed the
    generator without re-running `python scripts/generate_docs.py`."""
    for r in gd.build_reports(random.Random(gd.SEED)).values():
        assert (gd.DOCS_DIR / r["doc"]).read_text(encoding="utf-8") == r["text"], r["doc"]


def test_questions_are_valid_against_the_documents():
    reports = gd.build_reports(random.Random(gd.SEED))
    docs = {p.name: p.read_text(encoding="utf-8") for p in gd.DOCS_DIR.glob("ddr_*.md")}
    gd.validate(gd.build_questions(reports), docs)  # raises if evidence is missing or not unique
