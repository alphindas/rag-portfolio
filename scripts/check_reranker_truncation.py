"""
scripts/check_reranker_truncation.py

MEASUREMENT ONLY. It changes no code, index or result file.

bge-reranker-base reads at most 512 tokens for question + chunk. This script
  1. counts the rerank candidate pairs that are longer (top 10 of the
     BM25-weighted hybrid, as saved in eval/retrieved.json), checks which side
     gets cut, and whether gold evidence is in the cut part;
  2. compares, on the same candidates:
       A   current behaviour (the library truncates)
       B   long pairs split into overlapping windows that each fit in 512
           tokens with the question; chunk score = max over its windows
       B2  extra sensitivity check: like B, but the chunk's "[well | day |
           section]" line is repeated at the start of every later window
  3. writes eval/reranker_truncation_check.md (refuses to overwrite it unless
     --overwrite is given).

Uses the reranker on the GPU only. No Ollama, no Groq. Run from the repo root:
    HF_HUB_OFFLINE=1 PYTHONPATH=. python -B scripts/check_reranker_truncation.py
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from eval.retrieval_metrics import gold_ranks, hit_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from eval.scores import select_subset  # noqa: E402
from retrieve.rerank import RERANKER_MODEL_NAME, _get_reranker  # noqa: E402

LIMIT = 512
OVERLAP = 128          # chunk tokens shared by neighbouring windows
CANDIDATE_SETUP = "hybrid_weighted"
SAVED_SETUP = "hybrid_weighted_rerank"
OUT = ROOT / "eval" / "reranker_truncation_check.md"


def pair_ids(tok, question: str, text: str) -> list[int]:
    return tok(question, text, truncation=False, verbose=False)["input_ids"]


def kept_chars(tok, question: str, text: str) -> int:
    """How many characters of the chunk survive the library's truncation."""
    enc = tok(question, text, truncation="longest_first", max_length=LIMIT, return_offsets_mapping=True)
    return max(end for (_, end), seq in zip(enc["offset_mapping"], enc.sequence_ids()) if seq == 1)


def check_truncation(ce, question: str, text: str) -> dict:
    """What the model REALLY receives (CrossEncoder.preprocess) for a long pair."""
    tok = ce.tokenizer
    real = ce[0].preprocess([[question, text]])["input_ids"][0].tolist()
    q = tok(question, add_special_tokens=False)["input_ids"]
    d = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
    room = LIMIT - len(q) - tok.num_special_tokens_to_add(pair=True)
    s, e = tok.cls_token_id, tok.sep_token_id
    expected = [s] + q + [e, e] + d[:room] + [e]  # XLM-RoBERTa pair format: <s> q </s></s> d </s>
    return {"matches_chunk_tail_cut": real == expected, "length": len(real),
            "chunk_tokens_lost": len(d) - room, "question_tokens": len(q), "chunk_tokens": len(d)}


def windows(tok, question: str, text: str, repeat_header: bool) -> list[str]:
    """Substrings of `text`, each fitting in LIMIT together with the question.
    The first window is exactly what the current truncation keeps."""
    if len(pair_ids(tok, question, text)) <= LIMIT:
        return [text]
    offsets = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
    header = text.split("\n", 1)[0] + "\n"
    n_q = len(tok(question, add_special_tokens=False)["input_ids"])
    room = LIMIT - n_q - tok.num_special_tokens_to_add(pair=True)
    out, start = [], 0
    while True:
        prefix = header if (repeat_header and start > 0) else ""
        size = room - (len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0)
        while True:  # re-tokenising a substring can differ by a token: shrink until it fits
            end = min(start + size, len(offsets))
            piece = prefix + text[offsets[start][0]:offsets[end - 1][1]]
            if len(pair_ids(tok, question, piece)) <= LIMIT:
                break
            size -= 1
        out.append(piece)
        if end == len(offsets):
            return out
        start = end - OVERLAP


def rank_candidates(ce, question: str, candidates: list[dict], mode: str):
    """Returns (ids best-first, {id: score}, number of model inputs)."""
    if mode == "A":
        texts, owner = [c["text"] for c in candidates], list(range(len(candidates)))
    else:
        texts, owner = [], []
        for i, c in enumerate(candidates):
            for piece in windows(ce.tokenizer, question, c["text"], repeat_header=(mode == "B2")):
                texts.append(piece)
                owner.append(i)
    raw = ce.predict([[question, t] for t in texts], show_progress_bar=False)
    best = {}
    for i, score in zip(owner, raw):
        best[i] = max(best.get(i, float("-inf")), float(score))
    order = sorted(range(len(candidates)), key=lambda i: best[i], reverse=True)  # stable, like rerank()
    return ([candidates[i]["chunk_id"] for i in order],
            {candidates[i]["chunk_id"]: best[i] for i in order}, len(texts))


def metrics(questions: list[dict], ranking: dict) -> dict:
    gold = lambda q: [set(g) for g in q["gold_chunk_ids"]]
    avg = lambda f: sum(f(q) for q in questions) / len(questions)
    return {"Hit@2": avg(lambda q: hit_at_k(ranking[q["id"]], gold(q), 2)),
            "Hit@3": avg(lambda q: hit_at_k(ranking[q["id"]], gold(q), 3)),
            "Recall@2": avg(lambda q: recall_at_k(ranking[q["id"]], gold(q), 2)),
            "Recall@3": avg(lambda q: recall_at_k(ranking[q["id"]], gold(q), 3)),
            "MRR": avg(lambda q: reciprocal_rank(ranking[q["id"]], gold(q)))}


def short(chunk_id: str) -> str:
    return chunk_id.split("/")[-1].replace("ddr_", "").replace(".md", "")


def table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.overwrite:
        raise SystemExit(f"{OUT} exists (it may hold hand-written notes). Use --overwrite to replace it.")
    if not torch.cuda.is_available():
        raise SystemExit("No GPU available.")

    retrieved = json.loads((ROOT / "eval" / "retrieved.json").read_text())
    ce = _get_reranker()
    tok = ce.tokenizer
    md = ["# Reranker truncation check", "",
          f"Measurement only, from `scripts/check_reranker_truncation.py`. Model `{RERANKER_MODEL_NAME}` on "
          f"`{ce.device}`, input limit {ce.max_seq_length} tokens. Candidates: the top 10 of the BM25-weighted "
          "hybrid saved in `eval/retrieved.json`. No Ollama and no Groq calls.", ""]

    # ---- 1. counts ----
    over, total, mechanics = [], 0, []
    for q in retrieved:
        for pos, c in enumerate(q["rankings"][CANDIDATE_SETUP], 1):
            total += 1
            n = len(pair_ids(tok, q["user_input"], c["text"]))
            if n > LIMIT:
                over.append({"q": q, "c": c, "pos": pos, "tokens": n})
                mechanics.append(check_truncation(ce, q["user_input"], c["text"]))
    qids = sorted({o["q"]["id"] for o in over})
    chunks = sorted({short(o["c"]["chunk_id"]) for o in over})
    ok = sum(m["matches_chunk_tail_cut"] for m in mechanics)
    lost = [m["chunk_tokens_lost"] for m in mechanics]

    md += ["## 1. How many pairs are too long", "",
           f"- **{len(over)} of {total}** (question, candidate) pairs are longer than {LIMIT} tokens "
           f"(longest {max(o['tokens'] for o in over)}).",
           f"- **{len(qids)} of {len(retrieved)}** questions have at least one: {', '.join(qids)}.",
           f"- {len(chunks)} different chunks are involved: {', '.join(chunks)}.", "",
           "### Which part is cut", "",
           "The library uses `truncation=\"longest_first\"` on the right side: it removes tokens from the end of "
           "the longer input. Checked on what `CrossEncoder` really sends to the model: for "
           f"**{ok} of {len(over)}** long pairs the input equals *whole question + the start of the chunk*. "
           f"So **the end of the chunk is cut, never the question** (questions here are "
           f"{min(m['question_tokens'] for m in mechanics)}-{max(m['question_tokens'] for m in mechanics)} tokens, "
           f"these chunks {min(m['chunk_tokens'] for m in mechanics)}-{max(m['chunk_tokens'] for m in mechanics)}). "
           f"Chunk tokens lost per long pair: min {min(lost)}, median {statistics.median(lost):g}, max {max(lost)}.", ""]

    # is the cut text also in the document's next chunk (chunk overlap)?
    corpus = {c["chunk_id"]: c["text"] for q in retrieved for ranked in q["rankings"].values() for c in ranked}
    covered, partial, unknown = 0, [], 0
    for o in over:
        doc, index = o["c"]["chunk_id"].rsplit("::", 1)
        sibling = corpus.get(f"{doc}::{int(index) + 1}")
        tail = o["c"]["text"][kept_chars(tok, o["q"]["user_input"], o["c"]["text"]):].strip()
        if sibling is None:
            unknown += 1
        elif tail in sibling:
            covered += 1
        else:
            partial.append(short(o["c"]["chunk_id"]))
    md += ["### Is the cut text available elsewhere?", "",
           f"Long reports are split into two overlapping chunks. For **{covered} of {len(over)}** long pairs, all of "
           "the cut-off text also appears in the next chunk of the same report"
           + (f"; not fully for {len(partial)} ({', '.join(sorted(set(partial)))})" if partial else "")
           + (f"; next chunk not found in the saved rankings for {unknown}" if unknown else "")
           + ". This does not mean the next chunk is among the 10 candidates for that question.", ""]

    # gold evidence in the cut part
    rows, gold_chunk_pairs, affected = [], 0, []
    for o in over:
        q, c = o["q"], o["c"]
        items = [(i, e) for i, e in enumerate(q["evidence"]) if c["chunk_id"] in q["gold_chunk_ids"][i]]
        if not items:
            continue
        gold_chunk_pairs += 1
        keep = kept_chars(tok, q["user_input"], c["text"])
        for _, e in items:
            start = c["text"].find(e["text"])
            end = start + len(e["text"])
            state = "kept (fully visible)" if end <= keep else "CUT" if start >= keep else "PARTLY CUT"
            if end > keep:
                affected.append(q["id"])
            rows.append([q["id"], short(c["chunk_id"]), o["pos"], o["tokens"], f"`{e['text']}`",
                         f"{start}-{end} of {len(c['text'])}", keep, state])
    md += ["### Gold evidence and the cut-off part", "",
           f"{gold_chunk_pairs} long pairs involve a **gold chunk** for that question. What matters is whether "
           "the gold **phrase** is in the part that is cut:", ""]
    md += table(["Question", "Chunk", "Hybrid position", "Pair tokens", "Gold phrase", "Phrase characters",
                 "Characters kept", "Gold phrase"], rows)
    md += ["", f"**Pairs where the cut-off part contains gold evidence: {len(affected)}**"
           + (f" ({', '.join(affected)})." if affected else "."), ""]

    # ---- 2. experiment ----
    modes = ("A", "B", "B2")
    rank, scores, inputs, ms = {m: {} for m in modes}, {m: {} for m in modes}, {m: 0 for m in modes}, {m: [] for m in modes}
    for q in retrieved:  # warm-up pass so the first timing is not model setup
        rank_candidates(ce, q["user_input"], q["rankings"][CANDIDATE_SETUP], "A")
        break
    for q in retrieved:
        for m in modes:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rank[m][q["id"]], scores[m][q["id"]], n = rank_candidates(ce, q["user_input"], q["rankings"][CANDIDATE_SETUP], m)
            torch.cuda.synchronize()
            ms[m].append((time.perf_counter() - t0) * 1000)
            inputs[m] += n

    same_order = sum(rank["A"][q["id"]] == [c["chunk_id"] for c in q["rankings"][SAVED_SETUP]] for q in retrieved)
    max_diff = max(abs(scores["A"][q["id"]][c["chunk_id"]] - c["rerank_score"])
                   for q in retrieved for c in q["rankings"][SAVED_SETUP])
    lengthy = [len(w) for q in retrieved for c in q["rankings"][CANDIDATE_SETUP]
               for w in [windows(tok, q["user_input"], c["text"], False)] if len(w) > 1]
    lowered = sum(scores["B"][q][c] < scores["A"][q][c] - 1e-4 for q in scores["A"] for c in scores["A"][q])
    raised = sum(scores["B"][q][c] > scores["A"][q][c] + 1e-4 for q in scores["A"] for c in scores["A"][q])

    md += ["## 2. Retrieval experiment", "",
           "- **A**: current behaviour.",
           f"- **B**: long pairs split into windows that fit in {LIMIT} tokens with the question, neighbouring "
           f"windows sharing {OVERLAP} chunk tokens; chunk score = maximum over windows. Split pairs used "
           f"{min(lengthy)}-{max(lengthy)} windows.",
           "- **B2** (extra sensitivity check, not requested): B, with the chunk's `[well | day | section]` line "
           "repeated at the start of later windows, because a later window otherwise does not say which report "
           "it is from.",
           f"- **Check that A is faithful:** A reproduces the saved `{SAVED_SETUP}` order for "
           f"**{same_order} of {len(retrieved)}** questions (largest score difference {max_diff:.5f}).",
           f"- Because B's first window is what A already sees, B can raise a long chunk's score but should not "
           f"lower it. Observed over all {total} pairs: {raised} raised, {lowered} lowered.", ""]

    answerable = [q for q in retrieved if q["evidence"]]
    groups = [("All answerable", answerable),
              ("Hard", [q for q in answerable if q["category"] == "hard"]),
              ("Keyword holdout", [q for q in answerable if q["category"] == "keyword_holdout"])]
    for name, qs in groups:
        results = {m: metrics(qs, rank[m]) for m in modes}
        md += [f"### {name} (n={len(qs)})", ""]
        md += table(["Metric", "A (current)", "B (windows)", "B - A", "B2 (windows + header)", "B2 - A"],
                    [[k, f"{results['A'][k]:.4f}", f"{results['B'][k]:.4f}", f"{results['B'][k] - results['A'][k]:+.4f}",
                      f"{results['B2'][k]:.4f}", f"{results['B2'][k] - results['A'][k]:+.4f}"] for k in results["A"]])
        md += [""]

    # ---- 3. changes ----
    subset = {q["id"] for q in select_subset(retrieved, 30, 42)}
    for m in ("B", "B2"):
        changed = [q for q in retrieved if rank[m][q["id"]] != rank["A"][q["id"]]]
        top2 = [q["id"] for q in changed if rank[m][q["id"]][:2] != rank["A"][q["id"]][:2]]
        top2_set = [q["id"] for q in changed if set(rank[m][q["id"]][:2]) != set(rank["A"][q["id"]][:2])]
        rows, moves = [], {"up": [], "down": [], "unchanged": []}
        for q in changed:
            a, b = rank["A"][q["id"]], rank[m][q["id"]]
            moved = [f"{short(c)} {a.index(c) + 1}->{b.index(c) + 1}" for c in b if a.index(c) != b.index(c)]
            if q["evidence"]:
                gold = [set(g) for g in q["gold_chunk_ids"]]
                parts, directions = [], set()
                for n, (x, y) in enumerate(zip(gold_ranks(a, gold), gold_ranks(b, gold)), 1):
                    d = "unchanged" if x == y else "up" if (x is None or (y is not None and y < x)) else "down"
                    directions.add(d)
                    parts.append(f"evidence {n}: {x}->{y} ({d})")
                for d in ("up", "down"):
                    if d in directions:
                        moves[d].append(q["id"])
                if directions == {"unchanged"}:
                    moves["unchanged"].append(q["id"])
                gold_text = "; ".join(parts)
            else:
                gold_text = "no gold evidence (not documented)"
            rows.append([q["id"], q["category"], "yes" if q["id"] in top2 else "no",
                         "yes" if q["id"] in top2_set else "no", ", ".join(moved), gold_text])
        md += [f"## 3{'a' if m == 'B' else 'b'}. Questions whose top-10 order changes: A vs {m}", "",
               f"- **{len(changed)} of {len(retrieved)}** questions change somewhere in the order.",
               f"- Gold evidence moved **up** in {len(moves['up'])} ({', '.join(moves['up']) or 'none'}), "
               f"**down** in {len(moves['down'])} ({', '.join(moves['down']) or 'none'}), and kept its rank in "
               f"{len(moves['unchanged'])} ({', '.join(moves['unchanged']) or 'none'}).",
               f"- Top 2 (what the LLM receives) differs in order or content for **{len(top2)}**: "
               f"{', '.join(top2) or 'none'}. The **set** of two chunks differs for **{len(top2_set)}**: "
               f"{', '.join(top2_set) or 'none'}.",
               f"- Of the top-2 differences, in the 30-question RAGAS subset: "
               f"{', '.join(i for i in top2 if i in subset) or 'none'}.", ""]
        if rows:
            md += table(["Question", "Category", "Top-2 order changed", "Top-2 set changed",
                         "Chunks that moved (A rank->new rank)", "Gold evidence rank (A->new)"], rows) + [""]

    md += ["## Cost of windowing at query time", "",
           f"One timed pass per question on this GPU (n={len(retrieved)}, 10 candidates each, model already loaded; "
           "includes tokenising and splitting in Python). A rough figure, not a benchmark.", ""]
    md += table(["Setup", "Model inputs (all questions)", "Median ms per question", "Mean ms per question"],
                [[m, inputs[m], f"{statistics.median(ms[m]):.0f}", f"{statistics.mean(ms[m]):.0f}"] for m in modes])
    md += [""]
    OUT.write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
