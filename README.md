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
| Generation | `generate/answer.py` | Sends the chunks to a local Ollama model as numbered context and asks it to cite `[n]` for each claim. |
| Evaluation | `eval/` | Scores answers with RAGAS (see below). |

**Why plain Python?** No LangChain or LlamaIndex for the core pipeline. Every step is written out so it can be read and explained line by line.

---

## Evaluation

The eval is a two-step process, because the 4 GB GPU cannot hold the embedding model, the reranker and the LLM at the same time:

1. `eval/run_retrieval.py` runs search and reranking for every question and saves the results to `eval/retrieved.json`.
2. `eval/run_eval.py` generates answers (cached in `eval/generated.json`), then scores them with RAGAS:
   - **Faithfulness**: is the answer supported by the retrieved text?
   - **Response relevancy**: does the answer address the question?
   - **Context precision**: are the retrieved chunks relevant?
   - **Context recall**: did retrieval find what was needed?

Scores are saved metric by metric to `eval/results.csv`, so a rate-limit interruption does not lose finished work.

The test set (`eval/eval_set.json`) has 10 questions covering single-fact lookups, telling similar documents apart, a repeated question to test citation consistency, and one question whose correct answer is "not documented" to test hallucination.

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
- **Small corpus.** Five documents test cross-document retrieval but not splitting of long documents across chunks.

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

Build the indexes, then ask a question:

```bash
python ingest/embed.py
python ingest/bm25_index.py
python main.py
```

Run the evaluation:

```bash
python eval/run_retrieval.py
python eval/run_eval.py
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
