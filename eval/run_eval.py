# eval/run_eval.py
"""
Step 2 of the eval: reads eval/retrieved.json (from run_retrieval.py),
generates an answer for each question with the local LLM, then scores the
answers with RAGAS (Groq as judge).

A "run" is one retrieval setup plus how many chunks (k) go to the LLM, e.g.
"hybrid_rerank@3". By default we score the four runs the ablation needs:
hybrid and hybrid+reranker, each with k=2 and k=3. (BM25-only and
vector-only are compared on the free retrieval metrics only, to save the
Groq quota.)

Saved incrementally, so hitting a Groq quota wall loses nothing:
  - eval/generated.json: answers, {run: {question_id: {"answer": ...}}}.
    A rerun reuses them (local generation is slow).
  - eval/results.csv: one row per (run, question), one RAGAS metric column
    at a time. A rerun skips metrics already complete for a run.

Run AFTER `python -m eval.run_retrieval`:
    python -m eval.run_eval
    python -m eval.run_eval --runs hybrid_rerank@2     # just one run
"""
import argparse
import json
import os
import time

import ollama
import pandas as pd

from eval.ragas_judge import get_evaluator_llm, get_evaluator_embeddings

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from generate.answer import generate_answer
from generate.citations import check_citations
from observability.tracer import Trace

GENERATED_PATH = "eval/generated.json"
RESULTS_PATH = "eval/results.csv"
DEFAULT_RUNS = ["hybrid_weighted@2", "hybrid_weighted_rerank@2"]  # k=2 only: Groq free-tier quota

METRICS = {
    "faithfulness": Faithfulness,
    "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
    "llm_context_precision_without_reference": LLMContextPrecisionWithoutReference,
    "context_recall": LLMContextRecall,
}


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def chunks_for_run(question: dict, run: str) -> list[dict]:
    setup, k = run.split("@")
    return question["rankings"][setup][: int(k)]


def generate_with_retry(question: str, chunks: list[dict], attempts: int = 3) -> dict:
    """Ollama on a 4 GB laptop GPU occasionally dies with a one-off CUDA error
    ("device busy or unavailable"). Wait a few seconds and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return generate_answer(question, chunks)
        except ollama.ResponseError as e:
            if attempt == attempts:
                raise
            print(f"  Ollama error ({str(e)[:80]!r}), retrying in 10s ({attempt}/{attempts})")
            time.sleep(10)


def generate_for_run(retrieved: list[dict], run: str, generated: dict) -> dict:
    """Fills generated[run][question_id] for any question not answered yet."""
    answers = generated.setdefault(run, {})
    for i, q in enumerate(retrieved, start=1):
        if q["id"] in answers:
            continue
        print(f"[{run}] [{i}/{len(retrieved)}] generating: {q['user_input']}")
        chunks = chunks_for_run(q, run)
        trace = Trace(q["user_input"], source="eval_generation", run=run, question_id=q["id"])
        with trace.stage("generate"):
            result = generate_with_retry(q["user_input"], chunks)
        trace.finish(answer=result["answer"], citations=check_citations(result["answer"], len(chunks)))
        answers[q["id"]] = {"answer": result["answer"]}
        with open(GENERATED_PATH, "w") as f:  # save as we go
            json.dump(generated, f, indent=2)
    return answers


def build_dataset(retrieved: list[dict], run: str, answers: dict) -> EvaluationDataset:
    return EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=q["user_input"],
            response=answers[q["id"]]["answer"],
            retrieved_contexts=[c["text"] for c in chunks_for_run(q, run)],
            reference=q["reference"],
        )
        for q in retrieved
    ])


def load_results() -> pd.DataFrame:
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame(columns=["run", "id", "category", "user_input", "response", *METRICS])


def set_scores(results: pd.DataFrame, run: str, retrieved: list[dict], answers: dict,
               metric: str | None = None, scores: list | None = None) -> pd.DataFrame:
    """Makes sure every question of `run` has a row, and optionally writes one
    metric's scores (same order as `retrieved`). Rows are found by (run, id),
    never by question text, so a deliberately repeated question keeps 2 rows."""
    for q in retrieved:
        mask = (results["run"] == run) & (results["id"] == q["id"])
        if not mask.any():
            results.loc[len(results)] = {"run": run, "id": q["id"], "category": q["category"],
                                         "user_input": q["user_input"], "response": answers[q["id"]]["answer"]}
    if metric:
        for q, score in zip(retrieved, scores):
            results.loc[(results["run"] == run) & (results["id"] == q["id"]), metric] = score
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="comma-separated, e.g. hybrid@2,hybrid_rerank@3")
    parser.add_argument("--generate-only", action="store_true",
                        help="only generate answers with the local LLM; no Groq calls, no RAGAS scoring")
    args = parser.parse_args()
    runs = args.runs.split(",")

    import subprocess
    print("[gpu check before anything else]")
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv"], capture_output=True, text=True).stdout)

    retrieved = load_json("eval/retrieved.json", [])
    print(f"Loaded {len(retrieved)} retrieved results.")
    generated = load_json(GENERATED_PATH, {})

    # Generate everything first (local, no quota), then spend Groq quota.
    for run in runs:
        generate_for_run(retrieved, run, generated)

    if args.generate_only:
        print("Answers saved to eval/generated.json. Skipping RAGAS (--generate-only).")
        return

    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()
    run_config = RunConfig(max_workers=1, timeout=900, max_retries=8, max_wait=60)  # retries + backoff on rate limits

    results = load_results()
    for run in runs:
        answers = generated[run]
        results = set_scores(results, run, retrieved, answers)
        rows = results[results["run"] == run]
        for metric_name, metric_factory in METRICS.items():
            if rows[metric_name].notna().all():
                print(f"[{run}] skipping {metric_name} -- already scored")
                continue

            print(f"[{run}] scoring metric: {metric_name}")
            try:
                result = evaluate(
                    dataset=build_dataset(retrieved, run, answers),
                    metrics=[metric_factory()],
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                    run_config=run_config,
                    raise_exceptions=True,
                )
            except Exception as e:
                print(f"Stopped while scoring {metric_name} for {run}: {e}")
                print("Everything finished before this point is saved. Rerun later (e.g. after the quota resets):")
                print("it skips completed metrics and continues from here.")
                return
            results = set_scores(results, run, retrieved, answers, metric_name, result.to_pandas()[metric_name].tolist())
            results.to_csv(RESULTS_PATH, index=False)
            print(f"[{run}] saved {metric_name} to {RESULTS_PATH}")
            rows = results[results["run"] == run]

    results.to_csv(RESULTS_PATH, index=False)
    print("All runs scored. Now run: python -m eval.ablation")


if __name__ == "__main__":
    main()
