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

Both runs are scored by the same judge model (eval/scores.py: JUDGE_MODEL).
Order: each metric for ALL runs, then the next metric (faithfulness and answer
relevancy first), so a quota stop still leaves a fair comparison.

Saved after EVERY question, so hitting a Groq quota wall loses nothing:
  - eval/generated.json: answers, {run: {question_id: {"answer": ...}}}.
    A rerun reuses them (local generation is slow).
  - eval/ragas_scores.json: judge model, question ids and scores. A rerun
    skips everything already scored.
  - eval/results.csv: the same scores as a table.

Rate limits: a per-minute limit waits and retries; a daily limit stops the
script. Run the same command again after the quota resets.

Run AFTER `python -m eval.run_retrieval`:
    python -m eval.run_eval --subset 30
    python -m eval.run_eval --subset 30 --list        # show the subset and progress, no Groq calls
"""
import argparse
import json
import math
import os
import subprocess
import time

import ollama
from eval import scores as store_lib
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
DEFAULT_RUNS = ["hybrid_weighted@2", "hybrid_weighted_rerank@2"]  # k=2 only: Groq free-tier quota

MAX_RATE_LIMIT_WAITS = 6    # per question, for per-minute limits
MAX_FAILURES_IN_A_ROW = 3   # other errors: skip the question, but give up if they keep coming

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


def score_one(question: dict, run: str, answer: str, metric_name: str, llm, embeddings, run_config) -> float | None:
    """One metric for one question. None means the judge produced no score."""
    sample = SingleTurnSample(
        user_input=question["user_input"],
        response=answer,
        retrieved_contexts=[c["text"] for c in chunks_for_run(question, run)],
        reference=question["reference"],
    )
    result = evaluate(dataset=EvaluationDataset(samples=[sample]), metrics=[METRICS[metric_name]()],
                      llm=llm, embeddings=embeddings, run_config=run_config,
                      raise_exceptions=True, show_progress=False)
    score = float(result.to_pandas()[metric_name].iloc[0])
    return None if math.isnan(score) else score


def score_with_rate_limit_retry(*args) -> float | None:
    """Waits out per-minute limits. Daily limits and other errors are raised."""
    for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
        try:
            return score_one(*args)
        except Exception as e:
            if not store_lib.is_rate_limit(e) or store_lib.is_daily_limit(e) or attempt == MAX_RATE_LIMIT_WAITS:
                raise
            wait = store_lib.retry_wait_seconds(e)
            print(f"  rate limit, waiting {wait:.0f}s ({attempt + 1}/{MAX_RATE_LIMIT_WAITS})")
            time.sleep(wait)


def score_all(store: dict, runs: list[str], retrieved: list[dict], generated: dict, llm, embeddings, run_config) -> None:
    by_id = {q["id"]: q for q in retrieved}
    todo = store_lib.pending(store, runs)
    print(f"Judge: {store['judge_model']}. {len(todo)} scores to go.")
    failures = 0
    for i, (metric_name, run, qid) in enumerate(todo, start=1):
        try:
            score = score_with_rate_limit_retry(by_id[qid], run, generated[run][qid]["answer"],
                                                metric_name, llm, embeddings, run_config)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {metric_name} {run} {qid} failed: {e}")
            failures += 1
            if store_lib.is_rate_limit(e) or failures == MAX_FAILURES_IN_A_ROW:
                print("Stopping. Everything scored so far is saved. Run the same command again later")
                print("(after the quota resets): it continues from here.")
                return
            continue  # left unscored, so the next run tries it again
        failures = 0
        store["scores"].setdefault(run, {}).setdefault(metric_name, {})[qid] = score
        store_lib.save_store(store)
        store_lib.results_table(store, retrieved, generated).to_csv(store_lib.RESULTS_PATH, index=False)
        print(f"[{i}/{len(todo)}] {metric_name} {run} {qid}: {'no score' if score is None else f'{score:.2f}'}")
    left = len(store_lib.pending(store, runs))
    print(f"{left} scores still missing; run again." if left else "All runs scored. Now run: python -m eval.ablation")


def print_progress(store: dict, runs: list[str], subset: list[dict]) -> None:
    print(f"\nJudge: {store['judge_model']}. {len(subset)} questions (seed {store['seed']}):")
    for q in subset:
        print(f"  {q['id']} [{q['category']}] {q['user_input']}")
    print()
    for metric_name in METRICS:
        for run in runs:
            done = store["scores"].get(run, {}).get(metric_name, {})
            empty = sum(v is None for v in done.values())
            print(f"  {metric_name:42s} {run:26s} {len(done)}/{len(subset)}" + (f" ({empty} with no score)" if empty else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="comma-separated, e.g. hybrid@2,hybrid_rerank@3")
    parser.add_argument("--subset", type=int, help="score only this many questions, picked at random with --seed")
    parser.add_argument("--seed", type=int, default=store_lib.DEFAULT_SEED, help="seed for --subset (default %(default)s)")
    parser.add_argument("--list", action="store_true", help="show the questions and scoring progress, then exit")
    parser.add_argument("--generate-only", action="store_true",
                        help="only generate answers with the local LLM; no Groq calls, no RAGAS scoring")
    args = parser.parse_args()
    runs = args.runs.split(",")

    retrieved = load_json("eval/retrieved.json", [])
    subset = store_lib.select_subset(retrieved, args.subset, args.seed)
    store = store_lib.load_store([q["id"] for q in subset], args.seed)
    if args.list:
        print_progress(store, runs, subset)
        return

    print("[gpu check before anything else]")
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv"], capture_output=True, text=True).stdout)

    print(f"Loaded {len(retrieved)} retrieved results, using {len(subset)}.")
    generated = load_json(GENERATED_PATH, {})

    # Generate everything first (local, no quota), then spend Groq quota.
    for run in runs:
        generate_for_run(subset, run, generated)

    if args.generate_only:
        print("Answers saved to eval/generated.json. Skipping RAGAS (--generate-only).")
        return

    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()
    run_config = RunConfig(max_workers=1, timeout=900, max_retries=8, max_wait=60)  # retries + backoff on rate limits
    score_all(store, runs, retrieved, generated, evaluator_llm, evaluator_embeddings, run_config)


if __name__ == "__main__":
    main()
