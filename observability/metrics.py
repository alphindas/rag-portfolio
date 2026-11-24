"""
observability/metrics.py

Reads logs/traces.jsonl and prints latency percentiles per stage plus
citation statistics.

    python -m observability.metrics                  # all traces
    python -m observability.metrics --source cli     # only traces tagged source="cli"

Note: the first query in a process includes model loading time, so p95 on a
small log is dominated by those cold starts.
"""

import argparse
import json
from pathlib import Path

from observability.tracer import TRACE_PATH


def percentile(values: list[float], p: float) -> float:
    """p in 0..100, linear interpolation between neighbours."""
    values = sorted(values)
    pos = (len(values) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def load_traces(path: Path = TRACE_PATH, source: str | None = None) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        traces = [json.loads(line) for line in f if line.strip()]
    return [t for t in traces if source is None or t.get("source") == source]


def summarize(traces: list[dict]) -> dict:
    per_stage: dict[str, list[float]] = {}
    for t in traces:
        for s in t["stages"]:
            per_stage.setdefault(s["name"], []).append(s["ms"])
    stages = {name: {"n": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95)}
              for name, v in per_stage.items()}
    totals = [t["total_ms"] for t in traces]
    answered = [t["citations"] for t in traces if t.get("citations")]
    return {
        "queries": len(traces), "stages": stages,
        "total": {"p50": percentile(totals, 50), "p95": percentile(totals, 95)} if totals else None,
        "avg_citations": sum(c["n_cited"] for c in answered) / len(answered) if answered else None,
        "answers_checked": len(answered),
        "status_counts": {s: sum(c["status"] == s for c in answered) for s in ("ok", "declined", "no_citations", "invalid")},
    }


def print_report(summary: dict) -> None:
    print(f"Queries traced: {summary['queries']}\n")
    print("| Stage | n | p50 (ms) | p95 (ms) |\n|---|---|---|---|")
    for name in ("bm25", "vector", "fusion", "rerank", "generate"):
        if name in summary["stages"]:
            s = summary["stages"][name]
            print(f"| {name} | {s['n']} | {s['p50']:.1f} | {s['p95']:.1f} |")
    if summary["total"]:
        print(f"| **whole query** | {summary['queries']} | {summary['total']['p50']:.1f} | {summary['total']['p95']:.1f} |")
    if summary["avg_citations"] is not None:
        c = summary["status_counts"]
        print(f"\nAverage citations per answer: {summary['avg_citations']:.2f} ({summary['answers_checked']} answers)")
        print(f"Citation check: {c['ok']} ok, {c['declined']} declined, "
              f"{c['no_citations']} with no citations, {c['invalid']} citing a non-existent source")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help='only traces with this "source" tag (cli, app, eval_retrieval, eval_generation)')
    summary = summarize(load_traces(source=parser.parse_args().source))
    if not summary["queries"]:
        raise SystemExit(f"No traces found in {TRACE_PATH}. Ask a question with main.py first.")
    print_report(summary)


if __name__ == "__main__":
    main()
