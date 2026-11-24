"""Tracer, metrics and citation checker. No models needed."""

import json

import pytest

from generate.citations import check_citations
from observability.metrics import percentile, summarize, load_traces
from observability.tracer import NULL_TRACE, Trace
from chunk import context_header


def test_trace_writes_one_json_line_with_stages(tmp_path):
    path = tmp_path / "traces.jsonl"
    trace = Trace("q1", source="test")
    with trace.stage("bm25") as st:
        st.record([("a.md::0", 3.14159), ("b.md::1", 1.0)])
    with trace.stage("generate"):
        pass
    trace.finish(path=path, answer="x", citations={"status": "ok", "n_cited": 1, "invalid": []})
    trace.finish(path=path)  # a second query appends a second line
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert [s["name"] for s in entry["stages"]] == ["bm25", "generate"]
    assert entry["stages"][0]["top"][0] == {"id": "a.md::0", "score": 3.1416}
    assert entry["source"] == "test" and entry["total_ms"] >= 0


def test_stage_is_timed_even_if_it_raises():
    trace = Trace("q")
    with pytest.raises(ValueError):
        with trace.stage("rerank"):
            raise ValueError
    assert "rerank" in trace.timings()


def test_null_trace_is_a_noop():
    with NULL_TRACE.stage("bm25") as st:
        st.record([("a", 1.0)])


def test_percentiles_and_summary(tmp_path):
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    assert percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48)
    assert percentile([7], 95) == 7
    path = tmp_path / "t.jsonl"
    for ms, n in [(10, 2), (30, 0)]:
        Trace("q", source="s").finish(path=path)
        entry = json.loads(path.read_text().splitlines()[-1])
        entry.update(stages=[{"name": "bm25", "ms": ms}], citations={"status": "ok", "n_cited": n, "invalid": []})
        lines = path.read_text().splitlines()[:-1] + [json.dumps(entry)]
        path.write_text("\n".join(lines) + "\n")
    summary = summarize(load_traces(path, source="s"))
    assert summary["queries"] == 2 and summary["stages"]["bm25"]["p50"] == 20
    assert summary["avg_citations"] == 1.0


@pytest.mark.parametrize("answer,n,status", [
    ("Mud weight was 9.2 ppg [1].", 2, "ok"),
    ("Both agree [1][2]. Also [1, 2].", 2, "ok"),
    ("It was 9.2 ppg.", 2, "no_citations"),
    ("It was 9.2 ppg [3].", 2, "invalid"),
    ("It was 9.2 ppg [0].", 2, "invalid"),
    ("Not documented in the provided context.", 2, "declined"),
])
def test_citation_checker(answer, n, status):
    assert check_citations(answer, n)["status"] == status


def test_context_header_names_well_day_and_section():
    doc = "# Daily Drilling Report -- Condor-9\n## Report Info\nWell: Condor-9\nReport Date: Day 5\n"
    assert context_header(doc, "Daily Drilling Report -- Condor-9 > Mud Properties") == "[Condor-9 | Day 5 | Mud Properties]"
    assert context_header("# Widget Framework\ntext", "Widget Framework > Usage") == "[Widget Framework | Usage]"
