# Reranker truncation check

Measurement only, from `scripts/check_reranker_truncation.py`. Model `BAAI/bge-reranker-base` on `cuda:0`, input limit 512 tokens. Candidates: the top 10 of the BM25-weighted hybrid saved in `eval/retrieved.json`. No Ollama and no Groq calls.

## 1. How many pairs are too long

- **45 of 450** (question, candidate) pairs are longer than 512 tokens (longest 562).
- **28 of 45** questions have at least one: q01, q02, q03, q04, q06, q07, q12, q13, q14, q15, q16, q18, q20, q21, q22, q23, q26, q29, q33, q37, q38, q39, q40, q41, q42, q43, q44, q45.
- 8 different chunks are involved: condor9_day05::0, falcon3_day12::0, falcon3_day14::0, kestrel4_day17::0, kestrel4_day19::0, merlin8_day06::0, osprey5_day09::0, osprey5_day10::0.

### Which part is cut

The library uses `truncation="longest_first"` on the right side: it removes tokens from the end of the longer input. Checked on what `CrossEncoder` really sends to the model: for **45 of 45** long pairs the input equals *whole question + the start of the chunk*. So **the end of the chunk is cut, never the question** (questions here are 9-25 tokens, these chunks 506-534). Chunk tokens lost per long pair: min 7, median 32, max 50.

### Is the cut text available elsewhere?

Long reports are split into two overlapping chunks. For **45 of 45** long pairs, all of the cut-off text also appears in the next chunk of the same report. This does not mean the next chunk is among the 10 candidates for that question.

### Gold evidence and the cut-off part

6 long pairs involve a **gold chunk** for that question. What matters is whether the gold **phrase** is in the part that is cut:

| Question | Chunk | Hybrid position | Pair tokens | Gold phrase | Phrase characters | Characters kept | Gold phrase |
|---|---|---|---|---|---|---|---|
| q15 | falcon3_day12::0 | 7 | 524 | `Rate of penetration averaged 10.5 ft/hr` | 257-296 of 1708 | 1673 | kept (fully visible) |
| q21 | condor9_day05::0 | 8 | 562 | `averaged 19.1 ft/hr` | 276-295 of 1849 | 1679 | kept (fully visible) |
| q22 | kestrel4_day17::0 | 2 | 556 | `from 12,405 ft to 12,525 ft` | 196-223 of 1876 | 1701 | kept (fully visible) |
| q38 | falcon3_day12::0 | 6 | 526 | `averaged 10.5 ft/hr` | 277-296 of 1708 | 1667 | kept (fully visible) |
| q41 | condor9_day05::0 | 9 | 555 | `Mud weight: 9.5 ppg` | 1648-1667 of 1849 | 1695 | kept (fully visible) |
| q44 | kestrel4_day19::0 | 6 | 556 | `Mud weight: 11.9 ppg.` | 1669-1690 of 1894 | 1740 | kept (fully visible) |

**Pairs where the cut-off part contains gold evidence: 0**.

## 2. Retrieval experiment

- **A**: current behaviour.
- **B**: long pairs split into windows that fit in 512 tokens with the question, neighbouring windows sharing 128 chunk tokens; chunk score = maximum over windows. Split pairs used 2-2 windows.
- **B2** (extra sensitivity check, not requested): B, with the chunk's `[well | day | section]` line repeated at the start of later windows, because a later window otherwise does not say which report it is from.
- **Check that A is faithful:** A reproduces the saved `hybrid_weighted_rerank` order for **45 of 45** questions (largest score difference 0.00000).
- Because B's first window is what A already sees, B can raise a long chunk's score but should not lower it. Observed over all 450 pairs: 3 raised, 0 lowered.

### All answerable (n=40)

| Metric | A (current) | B (windows) | B - A | B2 (windows + header) | B2 - A |
|---|---|---|---|---|---|
| Hit@2 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Hit@3 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Recall@2 | 0.9625 | 0.9625 | +0.0000 | 0.9625 | +0.0000 |
| Recall@3 | 0.9875 | 0.9875 | +0.0000 | 0.9875 | +0.0000 |
| MRR | 0.9625 | 0.9625 | +0.0000 | 0.9625 | +0.0000 |

### Hard (n=9)

| Metric | A (current) | B (windows) | B - A | B2 (windows + header) | B2 - A |
|---|---|---|---|---|---|
| Hit@2 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Hit@3 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Recall@2 | 0.8333 | 0.8333 | +0.0000 | 0.8333 | +0.0000 |
| Recall@3 | 0.9444 | 0.9444 | +0.0000 | 0.9444 | +0.0000 |
| MRR | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |

### Keyword holdout (n=6)

| Metric | A (current) | B (windows) | B - A | B2 (windows + header) | B2 - A |
|---|---|---|---|---|---|
| Hit@2 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Hit@3 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Recall@2 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| Recall@3 | 1.0000 | 1.0000 | +0.0000 | 1.0000 | +0.0000 |
| MRR | 0.9167 | 0.9167 | +0.0000 | 0.9167 | +0.0000 |

## 3a. Questions whose top-10 order changes: A vs B

- **2 of 45** questions change somewhere in the order.
- Gold evidence moved **up** in 0 (none), **down** in 0 (none), and kept its rank in 2 (q01, q14).
- Top 2 (what the LLM receives) differs in order or content for **2**: q01, q14. The **set** of two chunks differs for **2**: q01, q14.
- Of the top-2 differences, in the 30-question RAGAS subset: q01, q14.

| Question | Category | Top-2 order changed | Top-2 set changed | Chunks that moved (A rank->new rank) | Gold evidence rank (A->new) |
|---|---|---|---|---|---|
| q01 | single_fact | yes | yes | condor9_day05::0 6->2, condor7_day05::0 2->3, condor9_day04::0 3->4, condor9_day07::0 4->5, harrier2_day04::0 5->6 | evidence 1: 1->1 (unchanged) |
| q14 | similar_wells | yes | yes | condor9_day05::0 3->2, condor9_day03::0 2->3 | evidence 1: 1->1 (unchanged) |

## 3b. Questions whose top-10 order changes: A vs B2

- **2 of 45** questions change somewhere in the order.
- Gold evidence moved **up** in 0 (none), **down** in 0 (none), and kept its rank in 2 (q14, q44).
- Top 2 (what the LLM receives) differs in order or content for **1**: q14. The **set** of two chunks differs for **1**: q14.
- Of the top-2 differences, in the 30-question RAGAS subset: q14.

| Question | Category | Top-2 order changed | Top-2 set changed | Chunks that moved (A rank->new rank) | Gold evidence rank (A->new) |
|---|---|---|---|---|---|
| q14 | similar_wells | yes | yes | condor9_day05::0 3->2, condor9_day03::0 2->3 | evidence 1: 1->1 (unchanged) |
| q44 | hard | no | no | kestrel4_day17::0 5->4, kestrel4_day17::1 4->5 | evidence 1: 3->3 (unchanged); evidence 2: 1->1 (unchanged) |

## Cost of windowing at query time

One timed pass per question on this GPU (n=45, 10 candidates each, model already loaded; includes tokenising and splitting in Python). A rough figure, not a benchmark.

| Setup | Model inputs (all questions) | Median ms per question | Mean ms per question |
|---|---|---|---|
| A | 450 | 338 | 323 |
| B | 495 | 380 | 364 |
| B2 | 495 | 379 | 365 |

## Recommendation: document only

*(Written by hand after the run. `--overwrite` would remove this section.)*

**On this evaluation set, windowing makes no measurable change.**

- Hit@2, Hit@3, Recall@2, Recall@3 and MRR are identical for A, B and B2, on all 40 answerable questions, the 9 hard ones and the 6 keyword holdout ones.
- No gold evidence changes rank in either direction.
- No gold phrase is in a cut-off part. All cut text (7 to 50 tokens per pair) is also in the next chunk of the same report. The likely reason is that the chunker's overlap (a target of 80 tokens, moved as whole paragraphs) is larger than what gets cut.

**It is not "no effect at all".** With B, the second chunk sent to the LLM would change for q01 and q14; with B2, only for q14. In these cases the gold chunk stays first. What changes is a non-gold second chunk: `condor9_day05::0`, whose end was being cut, moves up. Their current answers are correct. Generation was not run with the new chunks, so whether any answer would change is unknown.

**Limits of this result**

- Hit@2 and Hit@3 are already 1.00, so they could only have stayed the same or fallen. Recall and MRR had some room and did not move.
- 40 answerable questions on synthetic data, with one overlap setting (128 tokens).
- Only the BM25-weighted candidates were tested. The plain `hybrid_rerank` setup and the reranked rows of the 2x2 ablation (including chunks without headers) were not.
- The safety comes from today's chunk sizes and overlap. Longer questions in the app, longer chunks, less overlap or other documents could cut text that is nowhere else, or gold evidence. Nothing warns when it happens.

**What to document (suggested wording)**

> The reranker reads at most 512 tokens for question plus chunk, so the end of some long chunks is cut (45 of 450 candidate pairs, never the question). Splitting long pairs into windows did not change Hit@k, Recall@k or MRR on this evaluation set, and no gold phrase was in a cut part. This holds for these documents and this chunking only.

Revisit if the chunker, the documents or the reranker change.

**If it were fixed anyway (window the long pairs in `retrieve/rerank.py`)**

| Step | Cost |
|---|---|
| Code and a unit test | Small change |
| Query time on this GPU | Median about 340 ms -> 380 ms per question for reranking (one timed pass, rough) |
| Re-run retrieval, ablation and the 2x2 | GPU only, no Groq. Overwrites saved retrieval and result files; the untested reranked numbers could change |
| Regenerate affected answers | Local Ollama. With B: q01 and q14. The script skips cached answers, so these two entries must be removed first |
| Re-score faithfulness | **Needs Groq quota again.** The scorer also skips saved scores and does not notice a changed answer, so remove those entries first. Rough figures from the earlier pre-run estimate (176k for 60 question-setup scores): about 6k tokens for the 2 questions, or about 88k to re-score all 30 for the reranker setup |

Re-scoring only 2 questions would mix two judge runs, and the judge is noisy, so a small change in the average would not mean much.
