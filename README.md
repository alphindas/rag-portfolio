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

`eval/eval_set.json` has 45 questions (40 with gold evidence): single facts, look-alike wells, multi-hop (answer needs 2 chunks), "not documented" (tests hallucination), keyword-heavy (exact IDs and numbers), an `inference` question, one deliberately repeated question (for citation consistency), and two groups added later as checks on the first design (see below): 6 `keyword_holdout` and 9 `hard` questions (two wells or two days, look-alike numbers). Every answerable question carries gold evidence, so retrieval can be scored without an LLM.

The eval is a two-step process, because the 4 GB GPU cannot hold the embedding model, the reranker and the LLM at the same time:

1. `eval/run_retrieval.py` runs four setups for every question and saves the ranked chunks to `eval/retrieved.json`: **(a)** BM25 only, **(b)** vector only, **(c)** hybrid RRF, **(d)** hybrid + reranker, plus the BM25-weighted versions of (c) and (d).
2. `eval/ablation.py` computes the free retrieval metrics (hit rate@k, recall@k, MRR) for all setups, with no LLM, and writes `eval/ablation_results.csv` (the numbers from before headers/weighting are kept in `eval/ablation_results_baseline.csv`). `eval/ablate_headers.py` splits the gain between the two retrieval changes.
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
- **Plain RRF buried exact-ID matches.** BM25 alone finds exact IDs and numbers, but plain hybrid pushed them down. With k=60, neighbouring ranks score almost the same, so a modest BM25 weight (3, 5, even 10) changed nothing; it only worked from 30 up. The weight is now 100 for questions that contain an exact ID or number (reasoning: it has to exceed about 62). **The rule was designed while looking at the first 4 keyword questions.** To check for overfitting I added 6 new keyword questions afterwards and did not tune on them:

  | Hit / MRR | Plain hybrid | BM25-weighted hybrid |
  |---|---|---|
  | Design set (4 questions, rule fires on 4): Hit@2 / Hit@3 / MRR | 0.25 / 0.50 / 0.36 | 0.75 / 1.00 / 0.71 |
  | **Holdout (6 new questions, rule fires on 4): Hit@2 / Hit@3 / MRR** | 0.17 / 0.67 / 0.36 | **0.67 / 0.83 / 0.67** |

  So the gain holds up on new questions, but it is smaller than on the design set. The 2 holdout questions without an exact ID or number (a decimal ROP and a decimal mud weight) are not touched by the rule. Still only 10 questions in total.
- **Nothing told the model to say "I don't know".** With a "Not documented in the provided context." instruction, 4 of 5 unanswerable questions are declined correctly (before: 1 of 5 cleanly). The software-doc question about `on_click` is still answered with invented behaviour, because the 3B model sees related text and fills the gap.
- **Repeated questions differed between runs.** Ollama now runs at temperature 0 with a fixed seed; the repeated question gave 4 of 4 identical answers.

**Still open**

- **Small local model.** The machine has 4 GB of VRAM, so generation uses `llama3.2:3b`. Llama 3.1 8B ran out of memory, and a smaller model (`phi`) hallucinated badly. The 3B model sometimes leaves the `[n]` citation off; the citation checker flags these (see the eval output and `main.py` warnings). Missing-citation rate on the cached eval answers (45 questions per setup, k=2): **hybrid 2/45 (4.4%), hybrid + reranker 2/45 (4.4%)**. Proper "Not documented" declines (13 and 15) need no citation; leaving them out, the rates are 2/32 (6.2%) and 2/30 (6.7%). No answer cites a number that does not exist. This is under the 20% threshold set for adding a citation retry, so there is none.
- **Falcon-1 injury question is an inference, not a lookup.** Neither report says "no injuries" (Day 12: "no lost-time incidents", Day 13: "no incidents or near-misses"). The model says "no mention of anyone being injured" even when it is given both correct reports. Retrieval finds both chunks, so this is a reading weakness. It has its own category (`inference`) and was not tuned for.
- **Section labels are approximate.** A chunk's section label comes from its first block, so a chunk that spans several sections names only the first. Some citations name the wrong section.
- **The CLI reloads models on every question** (see [Latency](#latency)). The app avoids this by parking the models in RAM, but the LLM still has to unload after each answer so retrieval fits on the GPU, so generation always includes loading it.
- **Groq free tier is a hard cap.** The judge model allows 200,000 tokens per day, and scoring one run (30 questions, 4 metrics) uses about that much. `eval/run_eval.py` resumes where it stopped.
- **The eval set is small.** 45 questions (40 with gold chunks): one question moves a hit rate by 2.5 points, and results on subsets of 4-10 questions are only indicative.
- **Synthetic data.** The reports are generated from templates, so wording is far more regular than real field reports.

---

## What each retrieval change contributed

The 2x2 below isolates the two changes (context headers on chunks, BM25-weighted RRF) on the 40 answerable questions (`python -m eval.ablate_headers`). "Keyword Hit@3" covers all 10 keyword questions.

Without the reranker:

| Variant | Hit@2 | Hit@3 | Recall@3 | MRR | Keyword Hit@3 |
|---|---|---|---|---|---|
| neither | 0.65 | 0.70 | 0.59 | 0.65 | 0.30 |
| weighting only | 0.75 | 0.82 | 0.71 | 0.73 | 0.80 |
| headers only | 0.68 | 0.82 | 0.71 | 0.71 | 0.60 |
| **both** | **0.80** | **0.90** | **0.79** | **0.79** | **0.90** |

With the reranker:

| Variant | Hit@2 | Hit@3 | Recall@3 | MRR | Keyword Hit@3 |
|---|---|---|---|---|---|
| neither | 0.88 | 0.88 | 0.80 | 0.84 | 0.70 |
| weighting only | 0.95 | 0.95 | 0.88 | 0.91 | 1.00 |
| headers only | 0.95 | 0.95 | 0.94 | 0.93 | 0.80 |
| **both** | **1.00** | **1.00** | **0.99** | **0.96** | **1.00** |

Reading it: each change helps on its own, the gains roughly add up, and the reranker adds a large gain on top of either. Headers help most when the answer sits in the second chunk of a long report (Recall@3 0.80 -> 0.94 with the reranker); weighting helps most on exact-ID questions.

**Hard questions** (9 questions that need two wells or two days; Recall counts both pieces of evidence): plain hybrid reaches Recall@2 0.50 and Recall@3 0.61; hybrid + reranker reaches 0.83 and 0.94. At k=2 the reranker still misses some second pieces of evidence, so k=3 is safer for multi-hop questions.

---

## Latency

Cold and warm are measured separately (`python scripts/measure_latency.py`, p50 / p95 in ms, this laptop with a 4 GB GPU). A single p95 over all queries is dominated by model loading, so it would mislead.

| Stage | Cold: CLI, models reloaded every question (n=3) | Warm: app, models loaded once (n=10) |
|---|---|---|
| BM25 | 11 / 14 | 6 / 19 |
| Vector search | 2,808 / 4,685 | 60 / 155 |
| Rerank | 6,740 / 7,470 | 620 / 645 |
| Generate (LLM) | 9,567 / 9,730 | 7,257 / 9,267 |
| **Whole question** | **19,584 / 22,629** | **9,930 / 12,499** |

The embedding model and reranker take about 10 s to load, once (app startup). Warm retrieval is under 1 s. Generation dominates warm latency because Ollama unloads the LLM after each answer (it has to, so the next retrieval fits on the GPU), and the warm total also includes about 2 s of moving the models between RAM and GPU. Small samples: treat these as orders of magnitude.

---

## Demo app

```bash
streamlit run app.py          # needs Ollama running locally
```

Type a question to see the answer, the cited chunks in expandable boxes (cited ones open by default), and per-stage timing. The sidebar toggle turns the reranker off, so you can see it matter live: for "What was the mud weight on Condor-9 on Day 5?" the reranker on finds the right report; off, hybrid search alone returns the look-alike Condor-7 report and the model answers "Not documented".

The embedder and reranker are loaded once and kept in RAM; they move to the GPU only while retrieving, then step aside so the LLM fits in 4 GB (measured: both on the GPU use 1.4 GB, which leaves too little for Ollama).

<!-- SCREENSHOT PLACEHOLDER: add docs/app.png and replace this line with ![Streamlit app](docs/app.png) -->
*(Screenshot to be added.)*

### Docker

Ollama runs on the **host**, not in the container. The image holds the app, the models and the index (CPU-only PyTorch, so the embedder and reranker run on the CPU inside the container; the LLM still uses your GPU through the host's Ollama).

```bash
docker build -t rag-portfolio .
docker run --rm -p 8501:8501 --add-host=host.docker.internal:host-gateway rag-portfolio
# then open http://localhost:8501
```

On the host, Ollama must accept connections from the container: start it with `OLLAMA_HOST=0.0.0.0 ollama serve` (by default it only listens on localhost). `OLLAMA_HOST` inside the container defaults to `http://host.docker.internal:11434`. `.env` is excluded from the image.

---

## Results

> **Pending.** The final table (RAGAS: faithfulness, answer relevancy, context precision, context recall for hybrid and hybrid + reranker, k=2) is not written yet. It waits for the Groq daily quota; see `eval/run_eval.py`. The retrieval-only results above do not need Groq and are final for this corpus.

<!-- RESULTS-TABLE-PLACEHOLDER: replace this block with the RAGAS table from `python -m eval.ablation` -->

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

Latency and citation statistics from the trace log (`logs/traces.jsonl`, written by every question):

```bash
python -m observability.metrics
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
- [x] Streamlit demo with reranker toggle and per-stage timing
- [x] Dockerfile (Ollama on the host)
- [ ] RAGAS results table (waiting on Groq quota)
- [ ] Better chunk-level section labels

---

## Tech stack

Python, ChromaDB, `rank_bm25`, `sentence-transformers` (BGE embeddings and reranker), Ollama (`llama3.2:3b`), RAGAS, Groq (judge).
