# eval/run_eval.py
"""
Phase 2: reads eval/retrieved.json (from run_retrieval.py), generates an
answer for each question (caching to eval/generated.json so a rerun after
a Groq quota wall doesn't redo local generation), then scores results
with RAGAS one metric at a time -- saving each metric's scores to
eval/results.csv as soon as it finishes, so hitting a Groq daily-quota
wall partway through doesn't lose already-completed metrics. A rerun
skips any metric already present in results.csv.

Run AFTER `python3 -m eval.run_retrieval`:
    python3 -m eval.run_eval
"""
import json
import os

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

GENERATED_PATH = "eval/generated.json"
RESULTS_PATH = "eval/results.csv"

METRICS = {
    "faithfulness": Faithfulness,
    "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
    "llm_context_precision_without_reference": LLMContextPrecisionWithoutReference,
    "context_recall": LLMContextRecall,
}


def load_retrieved(path: str = "eval/retrieved.json") -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_or_generate_answers(retrieved: list[dict]) -> list[dict]:
    """Reuses eval/generated.json if it already has an answer for a given
    question, so a rerun doesn't redo local generation for work already done."""
    cached = {}
    if os.path.exists(GENERATED_PATH):
        with open(GENERATED_PATH, "r") as f:
            for item in json.load(f):
                cached[item["user_input"]] = item

    generated = []
    for i, item in enumerate(retrieved, start=1):
        if item["user_input"] in cached:
            print(f"[{i}/{len(retrieved)}] using cached answer: {item['user_input']}")
            generated.append(cached[item["user_input"]])
            continue

        print(f"[{i}/{len(retrieved)}] generating: {item['user_input']}")
        chunks = item["retrieved_chunks"]
        result = generate_answer(item["user_input"], chunks)
        generated.append({
            "user_input": item["user_input"],
            "reference": item["reference"],
            "retrieved_chunks": chunks,
            "answer": result["answer"],
        })

    with open(GENERATED_PATH, "w") as f:
        json.dump(generated, f, indent=2)

    return generated


def build_dataset(generated: list[dict]) -> EvaluationDataset:
    samples = []
    for item in generated:
        context_texts = [c["text"] for c in item["retrieved_chunks"]]
        samples.append(SingleTurnSample(
            user_input=item["user_input"],
            response=item["answer"],
            retrieved_contexts=context_texts,
            reference=item["reference"],
        ))
    return EvaluationDataset(samples=samples)


def load_completed_metrics() -> dict:
    """Reads results.csv if it exists, returning {metric_name: {user_input: score}}
    for metrics already fully scored -- so we can skip them on rerun."""
    if not os.path.exists(RESULTS_PATH):
        return {}
    df = pd.read_csv(RESULTS_PATH)
    completed = {}
    for metric_name in METRICS:
        if metric_name in df.columns and df[metric_name].notna().all():
            completed[metric_name] = dict(zip(df["user_input"], df[metric_name]))
    return completed


def main():
    import subprocess
    print("[gpu check before anything else]")
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv"], capture_output=True, text=True).stdout)

    retrieved = load_retrieved()
    print(f"Loaded {len(retrieved)} retrieved results.")
    generated = load_or_generate_answers(retrieved)
    dataset = build_dataset(generated)

    evaluator_llm = get_evaluator_llm()
    evaluator_embeddings = get_evaluator_embeddings()
    run_config = RunConfig(max_workers=1, timeout=900, max_retries=8, max_wait=60)

    completed = load_completed_metrics()
    all_scores = {name: completed[name] for name in completed}

    for metric_name, metric_factory in METRICS.items():
        if metric_name in completed:
            print(f"Skipping {metric_name} -- already fully scored in {RESULTS_PATH}")
            continue

        print(f"Scoring metric: {metric_name}")
        try:
            result = evaluate(
                dataset=dataset,
                metrics=[metric_factory()],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=run_config,
                raise_exceptions=True,
            )
        except Exception as e:
            print(f"Stopped while scoring {metric_name}: {e}")
            print("Already-completed metrics were saved before this point.")
            print(f"Rerun this script later -- it will skip {list(completed.keys())} "
                  f"and pick up starting from {metric_name}.")
            break

        df = result.to_pandas()
        all_scores[metric_name] = dict(zip(df["user_input"], df[metric_name]))
        _save_results(generated, all_scores)
        print(f"Saved {metric_name} to {RESULTS_PATH}")

    else:
        print("All metrics scored successfully.")


def _save_results(generated: list[dict], all_scores: dict):
    rows = []
    for item in generated:
        row = {"user_input": item["user_input"], "response": item["answer"]}
        for metric_name in METRICS:
            row[metric_name] = all_scores.get(metric_name, {}).get(item["user_input"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
