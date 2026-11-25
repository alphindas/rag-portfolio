"""
Estimates the Groq tokens needed to score the subset. Makes NO API calls: it
builds the real RAGAS prompts for the cached answers and counts their tokens.

    python -m eval.estimate_cost --subset 30

Input tokens are counted (o200k_base, the tokenizer family of gpt-oss). Output
tokens, which include the model's hidden reasoning, cannot be known beforehand:
OUTPUT_PER_CALL is an assumed low / expected / high range.
"""
import argparse
import math
import re

import tiktoken

from eval.run_eval import DEFAULT_RUNS, GENERATED_PATH, chunks_for_run, load_json
from eval.scores import DEFAULT_SEED, JUDGE_MODEL, METRIC_NAMES, select_subset

from ragas.metrics._answer_relevance import ResponseRelevanceInput, ResponseRelevancePrompt
from ragas.metrics._context_precision import QAC, ContextPrecisionPrompt
from ragas.metrics._context_recall import QCA, ContextRecallClassificationPrompt
from ragas.metrics._faithfulness import (NLIStatementInput, NLIStatementPrompt, StatementGeneratorInput,
                                         StatementGeneratorPrompt)

OUTPUT_PER_CALL = (250, 500, 1000)  # assumption: low, expected, high
CHAT_OVERHEAD = 80                  # assumption: tokens the chat format adds to each request
DAILY_LIMIT = 200_000


def prompts_for(metric: str, q: dict, run: str, answer: str) -> list[str]:
    """The prompt strings RAGAS would send for one question and one metric."""
    contexts = [c["text"] for c in chunks_for_run(q, run)]
    if metric == "faithfulness":
        # The judge writes the statements; the answer's sentences stand in for them here.
        statements = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
        return [StatementGeneratorPrompt().to_string(StatementGeneratorInput(question=q["user_input"], answer=answer)),
                NLIStatementPrompt().to_string(NLIStatementInput(context="\n".join(contexts), statements=statements))]
    if metric == "answer_relevancy":
        return [ResponseRelevancePrompt().to_string(ResponseRelevanceInput(response=answer))]
    if metric == "llm_context_precision_without_reference":  # one call per chunk
        return [ContextPrecisionPrompt().to_string(QAC(question=q["user_input"], context=c, answer=answer))
                for c in contexts]
    return [ContextRecallClassificationPrompt().to_string(
        QCA(question=q["user_input"], context="\n".join(contexts), answer=q["reference"]))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--subset", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--daily-limit", type=int, default=DAILY_LIMIT)
    args = parser.parse_args()
    runs = args.runs.split(",")
    subset = select_subset(load_json("eval/retrieved.json", []), args.subset, args.seed)
    generated = load_json(GENERATED_PATH, {})
    encoding = tiktoken.get_encoding("o200k_base")

    print(f"Judge: {JUDGE_MODEL}, {len(subset)} questions x {len(runs)} setups. "
          f"Assumed output per call (low/expected/high): {OUTPUT_PER_CALL}\n")
    print("| Metric (both setups) | Calls | Input tokens | Total low | Total expected | Total high |")
    print("|---|---|---|---|---|---|")
    running, cumulative = [0, 0, 0], {}
    for metric in METRIC_NAMES:
        prompts = [p for run in runs for q in subset
                   for p in prompts_for(metric, q, run, generated[run][q["id"]]["answer"])]
        inputs = sum(len(encoding.encode(p)) + CHAT_OVERHEAD for p in prompts)
        totals = [inputs + len(prompts) * out for out in OUTPUT_PER_CALL]
        running = [a + b for a, b in zip(running, totals)]
        cumulative[metric] = list(running)
        print(f"| {metric} | {len(prompts)} | {inputs:,} | " + " | ".join(f"{t:,}" for t in totals) + " |")
    print("| **All four** | | | " + " | ".join(f"**{t:,}**" for t in running) + " |")
    print(f"\nDays at {args.daily_limit:,} tokens/day, if no tokens are used elsewhere (low/expected/high):")
    for metric, totals in cumulative.items():
        print(f"  finished through {metric}: day " + " / ".join(str(math.ceil(t / args.daily_limit)) for t in totals))


if __name__ == "__main__":
    main()
