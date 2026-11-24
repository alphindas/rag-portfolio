"""
scripts/measure_latency.py

Cold vs warm latency, measured separately (the p95 in the trace log is mostly
model loading, so one number for both would mislead).

  cold: CLI behaviour. Every question reloads the embedder + reranker.
  warm: app behaviour. Models are loaded once up front and parked in RAM.

    python scripts/measure_latency.py

Uses Ollama and the GPU (embedder + reranker, then the LLM, never together).
Traces go to logs/traces.jsonl with source "latency_cold" / "latency_warm".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from observability.metrics import load_traces, summarize  # noqa: E402
from pipeline import run_query, warm_up  # noqa: E402

N_COLD, N_WARM = 3, 10
questions = [q["user_input"] for q in json.load(open("eval/eval_set.json"))
             if q["category"] in ("single_fact", "similar_wells", "hard")]

print(f"cold: {N_COLD} questions, models reloaded each time")
for q in questions[:N_COLD]:
    run_query(q, source="latency_cold", keep_models=False)

print(f"warm: loading models once...")
print(f"  model load (paid once): {warm_up():.1f} s")
for q in questions[N_COLD:N_COLD + N_WARM]:
    run_query(q, source="latency_warm", keep_models=True)

for source in ("latency_cold", "latency_warm"):
    s = summarize(load_traces(source=source)[-(N_COLD if source == "latency_cold" else N_WARM):])
    print(f"\n### {source} (n={s['queries']})\n")
    print("| Stage | p50 (ms) | p95 (ms) |\n|---|---|---|")
    for name in ("bm25", "vector", "fusion", "rerank", "generate"):
        if name in s["stages"]:
            print(f"| {name} | {s['stages'][name]['p50']:.0f} | {s['stages'][name]['p95']:.0f} |")
    print(f"| **whole query** | {s['total']['p50']:.0f} | {s['total']['p95']:.0f} |")
