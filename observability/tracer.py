"""
observability/tracer.py

A tiny tracer: time each stage of a query and write ONE JSON line per query
to logs/traces.jsonl.

    trace = Trace("What was the mud weight?", source="cli")
    with trace.stage("bm25") as st:
        results = search_bm25(...)
        st.record(results)          # keep the top chunk IDs + scores
    ...
    trace.finish(answer="...")      # appends the line to logs/traces.jsonl

Stages used by the pipeline: bm25, vector, fusion, rerank, generate.
Code that is not being traced receives NULL_TRACE, which does nothing, so
tracing never needs `if trace:` checks.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent.parent / "logs" / "traces.jsonl"
TOP_N = 5  # how many chunks to keep per stage


class _Stage:
    def __init__(self):
        self.top = None

    def record(self, ranked) -> None:
        """ranked: [(chunk_id, score), ...] or [{"chunk_id":..., "<x>_score":...}, ...]"""
        top = []
        for item in ranked[:TOP_N]:
            if isinstance(item, dict):
                score = item.get("rerank_score", item.get("fused_score"))
                item = (item["chunk_id"], score)
            top.append({"id": item[0], "score": None if item[1] is None else round(float(item[1]), 4)})
        self.top = top


class Trace:
    def __init__(self, question: str, **meta):
        self.question = question
        self.meta = meta
        self.stages: list[dict] = []
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        handle, t0 = _Stage(), time.perf_counter()
        try:
            yield handle
        finally:  # still timed if the stage raises
            record = {"name": name, "ms": round((time.perf_counter() - t0) * 1000, 1)}
            if handle.top is not None:
                record["top"] = handle.top
            self.stages.append(record)

    def timings(self) -> dict:
        """{stage name: milliseconds}, for showing in the CLI / app."""
        return {s["name"]: s["ms"] for s in self.stages}

    def finish(self, path: Path | None = None, **extra) -> dict:
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": self.question,
                 **self.meta, "total_ms": round((time.perf_counter() - self._start) * 1000, 1),
                 "stages": self.stages, **extra}
        path = Path(path or TRACE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


class _NullStage:
    def record(self, ranked) -> None:
        pass


class NullTrace:
    """Same interface as Trace, does nothing."""
    @contextmanager
    def stage(self, name: str):
        yield _NullStage()


NULL_TRACE = NullTrace()
