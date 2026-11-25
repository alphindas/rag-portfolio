# eval/ablation.py
"""
Builds the ablation table from eval/retrieved.json (step 1) and, if it
exists, eval/ragas_scores.json (RAGAS scores from run_eval.py). No models, no
API calls: safe to run any time.

    python -m eval.ablation

Prints markdown tables and writes eval/ablation_results.csv.
"""
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from generate.citations import check_citations
from eval.retrieval_metrics import doc_hit_at_k, gold_ranks, hit_at_k, recall_at_k, reciprocal_rank
from eval.scores import METRIC_NAMES as RAGAS_COLS, SCORES_PATH

SETUPS = {"bm25": "BM25 only", "vector": "Vector only", "hybrid": "Hybrid (RRF)",
          "hybrid_rerank": "Hybrid + reranker", "hybrid_weighted": "Hybrid, BM25-weighted",
          "hybrid_weighted_rerank": "Hybrid, BM25-weighted + reranker"}
BASELINE_PATH = "eval/ablation_results_baseline.csv"            # before context headers / weighting
BASELINE_CAT_PATH = "eval/ablation_by_category_baseline.csv"
CAT_PATH = "eval/ablation_by_category.csv"
KS = (2, 3)  # how many chunks go to the LLM
OUT_PATH = "eval/ablation_results.csv"


def mean(values):
    values = [v for v in values if v is not None and v == v]  # drop None / NaN
    return sum(values) / len(values) if values else None


def retrieval_scores(q: dict, setup: str, k: int) -> dict:
    chunks = q["rankings"][setup]
    ids = [c["chunk_id"] for c in chunks]
    gold = [set(g) for g in q["gold_chunk_ids"]]
    return {"hit": hit_at_k(ids, gold, k), "doc_hit": doc_hit_at_k([c["source"] for c in chunks], q["gold_docs"], k),
            "recall": recall_at_k(ids, gold, k), "rr": reciprocal_rank(ids, gold)}


def citation_stats(answers: dict, k: int) -> dict:
    """Citation check (in code, no LLM) over all generated answers of a run."""
    checks = [check_citations(a["answer"], k) for a in answers.values()]
    n = len(checks)
    rate = lambda *statuses: sum(c["status"] in statuses for c in checks) / n
    return {"cite_ok_rate": rate("ok", "declined"), "cite_none_rate": rate("no_citations"),
            "cite_invalid_rate": rate("invalid"), "avg_citations": sum(c["n_cited"] for c in checks) / n}


def ragas_means(store: dict | None, run: str) -> dict:
    """Average per metric, only once every question of the subset has been tried,
    so a half-scored run is never compared with a full one."""
    metrics = store["scores"].get(run, {}) if store else {}
    n = len(store["question_ids"]) if store else 0
    row = {col: mean(list(metrics[col].values())) if len(metrics.get(col, {})) == n and n else None
           for col in RAGAS_COLS}
    scored = any(v is not None for v in row.values())
    return {**row, "judge_model": store["judge_model"] if scored else None, "ragas_n": n if scored else None}


def build_rows(retrieved: list[dict], store: dict | None, generated: dict) -> list[dict]:
    answerable = [q for q in retrieved if q["evidence"]]  # "not documented" has no gold chunk
    rows = []
    for setup, label in SETUPS.items():
        for k in KS:
            s = [retrieval_scores(q, setup, k) for q in answerable]
            row = {"setup": label, "k": k, "n_questions": len(answerable),
                   "hit_at_k": mean([x["hit"] for x in s]), "doc_hit_at_k": mean([x["doc_hit"] for x in s]),
                   "recall_at_k": mean([x["recall"] for x in s]), "mrr": mean([x["rr"] for x in s])}
            answers = generated.get(f"{setup}@{k}")
            row.update(citation_stats(answers, k) if answers else
                       dict.fromkeys(["cite_ok_rate", "cite_none_rate", "cite_invalid_rate", "avg_citations"]))
            row.update(ragas_means(store, f"{setup}@{k}"))
            rows.append(row)
    return rows


def by_category(retrieved: list[dict], k: int = 3) -> list[dict]:
    """Hit@k per question category, one row per category."""
    groups = defaultdict(list)
    for q in retrieved:
        if q["evidence"]:
            groups[q["category"]].append(q)
    return [{"category": cat, "n": len(qs),
             **{label: mean([retrieval_scores(q, s, k)["hit"] for q in qs]) for s, label in SETUPS.items()}}
            for cat, qs in groups.items()]


def markdown_table(rows: list[dict], columns: dict) -> str:
    """columns: {dict key: header}. Floats shown to 2 decimals, missing as '-'."""
    def cell(v):
        return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


SUBSETS = {
    "Keyword, design set (the 4 questions the BM25 rule was designed on)": lambda q: q["category"] == "keyword",
    "Keyword, holdout (6 new questions, not used for design)": lambda q: q.get("holdout"),
    "Hard (two wells / two days / look-alike numbers)": lambda q: q["category"] == "hard",
}


def print_subsets(retrieved: list[dict]) -> None:
    """Hit@k / MRR for hand-picked groups of questions, plain vs weighted hybrid side by side."""
    from retrieve.hybrid import has_exact_token  # lazy: pulls in torch
    cols = ("hybrid", "hybrid_weighted", "hybrid_rerank", "hybrid_weighted_rerank")
    for title, pick in SUBSETS.items():
        qs = [q for q in retrieved if pick(q)]
        fired = sum(has_exact_token(q["user_input"]) for q in qs)
        print(f"\n### {title}: n={len(qs)}, weighting rule fires on {fired}\n")
        table = []
        for metric, k in (("hit", 2), ("hit", 3), ("recall", 2), ("recall", 3), ("rr", 0)):
            row = {"metric": "MRR" if metric == "rr" else f"{metric.capitalize()}@{k}"}
            for setup in cols:
                row[setup] = mean([retrieval_scores(q, setup, k or 3)[metric] for q in qs])
            table.append(row)
        print(markdown_table(table, {"metric": "Metric", **{s: SETUPS[s] for s in cols}}))


def print_before_after(rows: list[dict], cats: list[dict]) -> None:
    """Compares against the saved baseline (plain chunks, equal-weight RRF)."""
    if not os.path.exists(BASELINE_PATH):
        return
    before = {(r["setup"], r["k"]): r for r in pd.read_csv(BASELINE_PATH).to_dict("records")}
    table = []
    for r in rows:
        b = before.get((r["setup"], r["k"]))
        table.append({"setup": r["setup"], "k": r["k"],
                      "hit_b": b["hit_at_k"] if b else None, "hit_a": r["hit_at_k"],
                      "mrr_b": b["mrr"] if b else None, "mrr_a": r["mrr"]})
    print("\n### Before / after (before = plain chunks + equal-weight RRF; '-' = setup did not exist)\n")
    print(markdown_table(table, {"setup": "Setup", "k": "k", "hit_b": "Hit@k before", "hit_a": "Hit@k after",
                                 "mrr_b": "MRR before", "mrr_a": "MRR after"}))
    if os.path.exists(BASELINE_CAT_PATH):
        base = {r["category"]: r for r in pd.read_csv(BASELINE_CAT_PATH).to_dict("records")}
        out = []
        for c in cats:
            b = base.get(c["category"], {})
            out.append({"category": f"{c['category']} (n={c['n']})",
                        **{f"{lab} {tag}": (b.get(lab) if tag == "before" else c[lab])
                           for lab in ("Hybrid (RRF)", "Hybrid + reranker", "Hybrid, BM25-weighted") for tag in ("before", "after")}})
        cols = {"category": "Category (Hit@3)"}
        for lab in ("Hybrid (RRF)", "Hybrid + reranker", "Hybrid, BM25-weighted"):
            cols[f"{lab} before"], cols[f"{lab} after"] = f"{lab}: before", f"{lab}: after"
        print("\n### Hit@3 by category, before / after\n")
        print(markdown_table(out, cols))


def print_watch_list(retrieved: list[dict]) -> None:
    """For questions flagged "watch": where did the gold chunk(s) rank in each
    setup, and what did the LLM answer? Tells retrieval failures (gold chunk
    missing) apart from reading failures (gold chunk given, answer still wrong)."""
    generated = json.load(open("eval/generated.json")) if os.path.exists("eval/generated.json") else {}
    for q in (q for q in retrieved if q.get("watch")):
        print(f"\n**Watch: {q['user_input']}**  (gold evidence: {[e['text'] for e in q['evidence']]})")
        for setup in SETUPS:
            ids = [c["chunk_id"] for c in q["rankings"][setup]]
            print(f"- {SETUPS[setup]}: gold rank per evidence item = {gold_ranks(ids, [set(g) for g in q['gold_chunk_ids']])}")
        for run, answers in generated.items():
            if q["id"] in answers:
                print(f"- answer from `{run}`: {answers[q['id']]['answer'].strip()[:200]!r}")


def main():
    retrieved = json.load(open("eval/retrieved.json"))
    store = json.load(open(SCORES_PATH)) if os.path.exists(SCORES_PATH) else None

    generated = json.load(open("eval/generated.json")) if os.path.exists("eval/generated.json") else {}
    rows = build_rows(retrieved, store, generated)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    n = rows[0]["n_questions"]
    print(f"\n### Retrieval ablation ({n} answerable questions; 'not documented' ones have no gold chunk)\n")
    print(markdown_table(rows, {"setup": "Setup", "k": "k", "hit_at_k": "Hit@k", "doc_hit_at_k": "DocHit@k",
                                "recall_at_k": "Recall@k", "mrr": "MRR"}))
    print("\n### Hit@3 by question category\n")
    cats = by_category(retrieved)
    pd.DataFrame(cats).to_csv(CAT_PATH, index=False)
    print(markdown_table([{**c, "category": f"{c['category']} (n={c['n']})"} for c in cats],
                         {"category": "Category", **{s: s for s in SETUPS.values()}}))
    print_before_after(rows, cats)
    print_subsets(retrieved)

    scored = [r for r in rows if r["judge_model"]]
    if scored:
        print(f"\n### RAGAS (judge: {store['judge_model']}; same {len(store['question_ids'])} questions for every run; "
              "'-' = not scored, or not fully scored yet)\n")
        print(markdown_table(scored, {"setup": "Setup", "k": "k", "faithfulness": "Faithfulness",
                                      "answer_relevancy": "Answer relevancy",
                                      "llm_context_precision_without_reference": "Ctx precision",
                                      "context_recall": "Ctx recall"}))
    cited = [r for r in rows if r["cite_ok_rate"] is not None]
    if cited:
        print("\n### Citation check (in code; 'ok' = valid citations or a proper 'not documented' decline)\n")
        print(markdown_table(cited, {"setup": "Setup", "k": "k", "cite_ok_rate": "OK", "cite_none_rate": "No citations",
                                     "cite_invalid_rate": "Invalid [n]", "avg_citations": "Avg [n] per answer"}))
    print_watch_list(retrieved)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
