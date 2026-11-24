# rag-portfolio

A RAG ("ask my documents") system built from scratch in plain Python, with a focus on **measuring quality honestly** instead of just showing a demo.

It answers questions from a small set of drilling-report style documents, cites its sources, and is scored with an automated evaluation pipeline. Everything runs locally except the evaluation judge.

> **Status:** retrieval, reranking, cited generation and evaluation are working. Tracing/latency logging and CI regression gating are in progress (see [Roadmap](#roadmap)).

---

## How it works

```
Documents -> Chunk -> Embed (BGE) -> Chroma vector index
                  \-> BM25 keyword index

Question -> Hybrid retrieval (BM25 + vectors, fused with RRF)
         -> Cross-encoder reranker (BGE)
         -> Local LLM answers with [n] citations
```

| Step | File | What it does |
|---|---|---|
| Chunking | `ingest/chunk.py` | Splits markdown by headers and paragraphs, then packs into 480-token chunks with 80-token overlap. Keeps a header breadcrumb such as `Installation > Requirements`. |
| Embedding | `ingest/embed.py` | Embeds chunks with `BAAI/bge-base-en` and stores them in ChromaDB (cosine). Queries and documents are encoded separately. |
| Keyword index | `ingest/bm25_index.py` | Builds a BM25 index. Chunk IDs match the vector index so the two result lists can be fused. |
| Hybrid search | `retrieve/hybrid.py` | Runs BM25 and vector search separately (top 20 each), then fuses with Reciprocal Rank Fusion (k=60). Uses ranks only, because BM25 and cosine scores are not comparable. |
| Reranking | `retrieve/rerank.py` | `BAAI/bge-reranker-base` re-scores the top 10 candidates together with the question and keeps the best. |
| Pipeline | `pipeline.py`, `main.py` | Chains search, rerank and generation, and frees the embedding model and reranker from GPU memory before the LLM loads. `main.py` is the command-line interface. |
| Generation | `generate/answer.py` | Sends the chunks to a local Ollama model as numbered context and asks it to cite `[n]` for each claim. |
| Evaluation | `eval/` | Scores answers with RAGAS (see below). |

**Why plain Python?** No LangChain or LlamaIndex for the core pipeline. Every step is written out so it can be read and explained line by line.

---

## Evaluation

The test corpus is **synthetic**: 40 fictional daily drilling reports (8 wells) plus one unrelated software doc, made by `scripts/generate_docs.py` with a fixed seed. It is built to be hard: look-alike wells (Condor-7/Condor-9, Falcon-1/Falcon-3), values that change every day, near-duplicate sentences, and 10 long reports that split into two chunks.

`eval/eval_set.json` has 30 questions: single facts, look-alike wells, multi-hop (answer needs 2 chunks), "not documented" (tests hallucination), keyword-heavy (exact IDs and numbers), and one deliberately repeated question (for citation consistency). Every question carries gold evidence, so retrieval can be scored without an LLM.

The eval is a two-step process, because the 4 GB GPU cannot hold the embedding model, the reranker and the LLM at the same time:

1. `eval/run_retrieval.py` runs four setups for every question and saves the ranked chunks to `eval/retrieved.json`: **(a)** BM25 only, **(b)** vector only, **(c)** hybrid RRF, **(d)** hybrid + reranker.
2. `eval/ablation.py` computes the free retrieval metrics (hit rate@k, recall@k, MRR) for all four setups, with no LLM, and writes `eval/ablation_results.csv`.
3. `eval/run_eval.py` generates answers for setups (c) and (d) with k=2 and k=3 chunks (cached in `eval/generated.json`), then scores them with RAGAS, one metric at a time into `eval/results.csv`:
   - **Faithfulness**: is the answer supported by the retrieved text?
   - **Response relevancy**: does the answer address the question?
   - **Context precision**: are the retrieved chunks relevant?
   - **Context recall**: did retrieval find what was needed?

Scores are saved metric by metric, so a rate-limit interruption does not lose finished work. Re-run `python -m eval.ablation` after any step to refresh the tables.

**Judge model:** Groq's free tier (RAGAS needs an LLM to grade answers), with local BGE embeddings for the relevancy metric.

Results: see `eval/results.csv`.

<!-- TODO: paste the latest average scores per metric here once you are happy with a run -->

---

## Known limitations and findings

These are written down on purpose. A project that hides its problems cannot be trusted.

- **Small local model.** The machine has 4 GB of VRAM, so generation uses `llama3.2:3b`. Llama 3.1 8B ran out of memory, and a smaller model (`phi`) hallucinated badly. The 3B model follows citation instructions inconsistently from run to run.
- **Citation enforcement is only a prompt.** There is no code that checks citations yet, so citation accuracy is measured by the eval and not assumed.
- **A retrieval setting hid a problem.** With a 5-chunk test corpus and `top_k=5`, every question retrieved the whole corpus, which gave a false-perfect context recall of 1.0 and let the model mix up facts between wells. Lowering `top_k` to 2 fixed the mix-ups. The lesson: a tiny corpus can make retrieval metrics look better than they are.
- **Misleading section labels.** `header_path` comes from the first block in a chunk, so a citation can name the wrong section when a short document stays in one chunk. Logged, not yet fixed.
- **Odd faithfulness scores.** Two simple, correct answers scored 0.5 on faithfulness while harder answers scored 1.0. This is suspected to be a quirk in how the judge splits statements, and it has not been confirmed.
- **Citation instability needs more than one pass.** A repeated question gave identical output in one automated run, which differs from earlier manual runs. One eval pass is not enough to measure this.
- **Groq free tier is a hard cap.** The judge model allows 200,000 tokens per day, and scoring one run (30 questions, 4 metrics) uses about that much. A full RAGAS comparison of four runs takes several days of quota. `eval/run_eval.py` resumes where it stopped.
- **Chunks lose their context.** In a report that splits in two, the second chunk (mud, downtime, safety) has no well name or day in its text, and the index only sees the text. Questions whose answer sits there are the ones retrieval misses most.
- **Equal-weight RRF dilutes exact matches.** On keyword questions BM25 alone finds the right chunk every time, but plain hybrid does not (see the ablation).
- **The model answers "not documented" questions unreliably.** On the unrelated software-doc question it describes behaviour that is not in the documents, in every run.
- **Falcon-1 injury question.** The model says "no mention of anyone being injured" even when it is given both correct reports (Day 12 and 13). Neither report literally says "no injuries" (they say "no lost-time incidents" and "no incidents or near-misses"), so the reference answer is an inference and the model's answer is arguably literal. Retrieval is not the problem here.
- **Repeated questions are not stable.** The same question with the same chunks gave identical answers in only one of four runs (the local model samples with a non-zero temperature).

---

## Data

The sample documents are **synthetic** (a fictional widget framework doc and made-up daily drilling reports). No real client or well data is in this repository.

---

## Setup

Requires Python 3.10+, [Ollama](https://ollama.com), and ideally an NVIDIA GPU.

```bash
git clone https://github.com/alphindas/rag-portfolio.git
cd rag-portfolio
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2:3b
```

For evaluation, create a `.env` file with your free Groq API key:

```
GROQ_API_KEY=your_key_here
```

Generate the synthetic documents and eval questions (already committed, so this is optional; same seed gives the same files):

```bash
python scripts/generate_docs.py
```

Build the indexes (reads every `.md` file in `sample_docs/`), then ask a question:

```bash
python ingest/build_index.py
python main.py                                   # interactive
python main.py --question "What was the mud weight on Condor-7 on Day 4?"
```

Run the tests (no GPU, Ollama or API key needed):

```bash
pytest
```

Run the evaluation (run from the repo root):

```bash
python -m eval.run_retrieval     # GPU: embedder + reranker, then exits
python -m eval.ablation          # free retrieval metrics, no LLM
python -m eval.run_eval          # Ollama generation + Groq-judged RAGAS (needs GROQ_API_KEY)
python -m eval.ablation          # table again, now including RAGAS
```

---

## Roadmap

- [x] Chunking, embeddings, BM25 index
- [x] Hybrid retrieval with RRF fusion
- [x] Cross-encoder reranking
- [x] Local LLM generation with citations
- [x] RAGAS evaluation pipeline
- [ ] Tracing and latency logging (p50 / p95), cost per request
- [ ] GitHub Actions: run the eval on every push and fail on quality regression
- [ ] Code-level citation checker
- [ ] Larger corpus and better chunk-level section labels

---

## Tech stack

Python, ChromaDB, `rank_bm25`, `sentence-transformers` (BGE embeddings and reranker), Ollama (`llama3.2:3b`), RAGAS, Groq (judge).
