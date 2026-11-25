"""
Light helpers for the RAGAS step (no ragas / torch imports, so they are cheap
to test and eval/ablation.py can use them): the judge model name, the fixed
question subset, the per-question score store and rate-limit detection.
"""
import json
import os
import random
import re

import pandas as pd

JUDGE_MODEL = "openai/gpt-oss-20b"  # Groq. Every run is scored with this one model.
DEFAULT_SEED = 42
SCORES_PATH = "eval/ragas_scores.json"  # source of truth for resume
RESULTS_PATH = "eval/results.csv"       # same scores as a table, rewritten from the store

# Order matters: each metric is scored for ALL runs before the next one starts.
METRIC_NAMES = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference", "context_recall"]


def select_subset(questions: list[dict], n: int | None, seed: int = DEFAULT_SEED) -> list[dict]:
    """A seeded random sample of n questions, kept in their original order.
    Same questions + same n + same seed -> same subset."""
    if n is None or n >= len(questions):
        return list(questions)
    picked = set(random.Random(seed).sample(range(len(questions)), n))
    return [q for i, q in enumerate(questions) if i in picked]


def load_store(question_ids: list[str], seed: int, path: str = SCORES_PATH) -> dict:
    """Loads saved scores. Refuses to continue if they came from another judge
    model or another question set: those averages must not be mixed."""
    if not os.path.exists(path):
        return {"judge_model": JUDGE_MODEL, "seed": seed, "question_ids": question_ids, "scores": {}}
    with open(path) as f:
        store = json.load(f)
    if store["judge_model"] != JUDGE_MODEL:
        raise SystemExit(f"{path} was scored by {store['judge_model']}, the judge is now {JUDGE_MODEL}. "
                         f"Move {path} away to start again.")
    if store["question_ids"] != question_ids:
        raise SystemExit(f"{path} holds {len(store['question_ids'])} other questions (seed {store['seed']}). "
                         f"Use the same --subset and --seed, or move {path} away to start again.")
    return store


def save_store(store: dict, path: str = SCORES_PATH) -> None:
    """Write-then-rename, so Ctrl+C in the middle cannot corrupt the file."""
    with open(path + ".tmp", "w") as f:
        json.dump(store, f, indent=2)
    os.replace(path + ".tmp", path)


def pending(store: dict, runs: list[str]) -> list[tuple[str, str, str]]:
    """(metric, run, question_id) still to score, in scoring order. A saved null
    (the judge produced no score) counts as done, so it is not paid for twice."""
    return [(m, run, qid) for m in METRIC_NAMES for run in runs for qid in store["question_ids"]
            if qid not in store["scores"].get(run, {}).get(m, {})]


def results_table(store: dict, retrieved: list[dict], generated: dict) -> pd.DataFrame:
    """One row per (run, question) for every run that has at least one score."""
    by_id = {q["id"]: q for q in retrieved}
    rows = [{"judge_model": store["judge_model"], "run": run, "id": qid, "category": by_id[qid]["category"],
             "user_input": by_id[qid]["user_input"], "response": generated[run][qid]["answer"],
             **{m: metrics.get(m, {}).get(qid) for m in METRIC_NAMES}}
            for run, metrics in store["scores"].items() for qid in store["question_ids"]]
    return pd.DataFrame(rows)


def is_rate_limit(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return "ratelimit" in text or "rate_limit" in text or "rate limit" in text or "error code: 429" in text


def is_daily_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "per day" in text or "(tpd)" in text or "(rpd)" in text


def retry_wait_seconds(error: Exception, default: float = 60.0) -> float:
    """Reads Groq's "Please try again in 1m2.5s" hint; adds 1 s of margin."""
    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)(ms|s))?", str(error))
    if not match or not any(match.groups()):
        return default
    hours, minutes, value, unit = match.groups()
    seconds = float(value or 0) / (1000 if unit == "ms" else 1)
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + seconds + 1
