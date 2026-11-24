# rag-portfolio

A RAG ("ask my documents") system built from scratch in plain Python, with a focus on **measuring quality honestly** instead of just showing a demo.

It answers questions from a small set of drilling-report style documents, cites its sources, and is scored with an automated evaluation pipeline. Everything runs locally except the evaluation judge.

> **Status:** retrieval, reranking, cited generation and evaluation are working. Tracing, latency metrics and a code-level citation checker are working; CI regression gating is still to do (see [Roadmap](#roadmap)).

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
| Chunking | `ingest/chunk.py` | Splits markdown by headers and paragraphs, then packs into 480-token chunks with 80-token overlap. Keeps a header breadcrumb such as `Installation > Requirements`, and puts a one-line label (`[Condor-9 \| Day 5 \| Mud Properties]`) at the top of every chunk so a chunk from the middle of a long report still says which report it belongs to. |
| Embedding | `ingest/embed.py` | Embeds chunks with `BAAI/bge-base-en` and stores them in ChromaDB (cosine). Queries and documents are encoded separately. |
| Keyword index | `ingest/bm25_index.py` | Builds a BM25 index. Chunk IDs match the vector index so the two result lists can be fused. |
| Hybrid search | `retrieve/hybrid.py` | Runs BM25 and vector search separately (top 20 each), then fuses with Reciprocal Rank Fusion (k=60). Uses ranks only, because BM25 and cosine scores are not comparable. When the question contains an exact ID or number (`WO-98389`, `10,955`), BM25 gets weight 100 (see the ablation for why it has to be that large). |
| Reranking | `retrieve/rerank.py` | `BAAI/bge-reranker-base` re-scores the top 10 candidates together with the question and keeps the best. |
| Pipeline | `pipeline.py`, `main.py` | Chains search, rerank and generation, and frees the embedding model and reranker from GPU memory before the LLM loads. `main.py` is the command-line interface. |
| Generation | `generate/answer.py` | Sends the chunks to a local Ollama model as numbered context and asks it to cite `[n]` for each claim. |
| Tracing | `observability/` | `tracer.py` times each stage (bm25, vector, fusion, rerank, generate) and writes one JSON line per query to `logs/traces.jsonl`. `python -m observability.metrics` prints p50/p95 per stage. `generate/citations.py` checks `[n]` citations in code. |
| Evaluation | `eval/` | Scores answers with RAGAS (see below). |

**Why plain Python?** No LangChain or LlamaIndex for the core pipeline. Every step is written out so it can be read and explained line by line.

---

## Evaluation

The test corpus is **synthetic**: 40 fictional daily drilling reports (8 wells) plus one unrelated software doc, made by `scripts/generate_docs.py` with a fixed seed. It is built to be hard: look-alike wells (Condor-7/Condor-9, Falcon-1/Falcon-3), values that change every day, near-duplicate sentences, and 10 long reports that split into two chunks.

`eval/eval_set.json` has 30 questions: single facts, look-alike wells, multi-hop (answer needs 2 chunks), "not documented" (tests hallucination), keyword-heavy (exact IDs and numbers), and one deliberately repeated question (for citation consistency). Every question carries gold evidence, so retrieval can be scored without an LLM.

The eval is a two-step process, because the 4 GB GPU cannot hold the embedding model, the reranker and the LLM at the same time:

1. `eval/run_retrieval.py` runs four setups for every question and saves the ranked chunks to `eval/retrieved.json`: **(a)** BM25 only, **(b)** vector only, **(c)** hybrid RRF, **(d)** hybrid + reranker, plus the BM25-weighted versions of (c) and (d).
2. `eval/ablation.py` computes the free retrieval metrics (hit rate@k, recall@k, MRR) for all four setups, with no LLM, and writes `eval/ablation_results.csv`.
3. `eval/run_eval.py` generates answers for the two hybrid setups (BM25-weighted, with and without reranker) with k=2 chunks (`--generate-only` stops before any Groq call) (cached in `eval/generated.json`), then scores them with RAGAS, one metric at a time into `eval/results.csv`:
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

**Found by the eval and fixed (kept here so the numbers make sense)**

- **Chunks lost their context.** The second chunk of a long report (mud, downtime, safety) had no well name or day, so neither BM25 nor the embeddings could tell which report it came from. Fixed by putting a header on every chunk. Hit@2 for hybrid + reranker went from 0.84 to 0.96.
- **Plain RRF buried exact-ID matches.** On the 4 keyword questions BM25 alone scored 1.00 but plain hybrid only 0.25. With k=60, neighbouring ranks score almost the same, so a modest BM25 weight (3, 5, even 10) changed nothing; it only worked at 30 and above. The weight is set to 100 for questions containing an exact ID or number: keyword Hit@3 went from 0.50 (with headers) to 1.00. Caveat: only 4 questions, and the rule for "exact ID or number" was written while looking at them.
- **Nothing told the model to say "I don't know".** With a "Not documented in the provided context." instruction, 4 of 5 unanswerable questions are declined correctly (before: 1 of 5 cleanly). The software-doc question about `on_click` is still answered with invented behaviour, because the 3B model sees related text and fills the gap.
- **Repeated questions differed between runs.** Ollama now runs at temperature 0 with a fixed seed; the repeated question gave 4 of 4 identical answers.

**Still open**

- **Small local model.** The machine has 4 GB of VRAM, so generation uses `llama3.2:3b`. Llama 3.1 8B ran out of memory, and a smaller model (`phi`) hallucinated badly. The 3B model often leaves the `[n]` citation off short answers; the citation checker flags these (see the eval output and `main.py` warnings).
- **Falcon-1 injury question is an inference, not a lookup.** Neither report says "no injuries" (Day 12: "no lost-time incidents", Day 13: "no incidents or near-misses"). The model says "no mention of anyone being injured" even when it is given both correct reports. Retrieval finds both chunks, so this is a reading weakness. It has its own category (`inference`) and was not tuned for.
- **Section labels are approximate.** A chunk's section label comes from its first block, so a chunk that spans several sections names only the first. Some citations name the wrong section.
- **Latency is mostly model loading.** To fit in 4 GB the embedder and reranker are dropped before the LLM loads, so every question reloads them (about 5-8 s each in the CLI). Warm, retrieval takes under 1 s; generation takes about 8-10 s.
- **Groq free tier is a hard cap.** The judge model allows 200,000 tokens per day, and scoring one run (30 questions, 4 metrics) uses about that much. `eval/run_eval.py` resumes where it stopped.
- **The eval set is small.** 30 questions (25 with gold chunks): one question moves a hit rate by 4 points, so differences under about 8 points are noise.
- **Synthetic data.** The reports are generated from templates, so wording is far more regular than real field reports.

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
- [x] Tracing and latency logging (p50 / p95)
- [ ] GitHub Actions: run the eval on every push and fail on quality regression
- [x] Code-level citation checker
- [x] Larger, harder synthetic corpus with gold evidence
- [x] Context headers on chunks, BM25-weighted fusion for exact IDs
- [ ] Better chunk-level section labels

---

## Tech stack

Python, ChromaDB, `rank_bm25`, `sentence-transformers` (BGE embeddings and reranker), Ollama (`llama3.2:3b`), RAGAS, Groq (judge).
