"""
pipeline.py

The full question -> answer path in one function, shared by main.py (and
later app.py and the eval scripts):

    hybrid search -> rerank -> free GPU memory -> generate

Why free memory in the middle? The 4 GB GPU cannot hold the embedding
model, the reranker AND the LLM together. The first two are only needed
for retrieval, so we drop them before Ollama loads the LLM.
"""

import gc
from pathlib import Path

from retrieve.hybrid import hybrid_search  # also puts ingest/ on sys.path
import retrieve.rerank as rerank_module
import embed as embed_module  # the same module object hybrid.py uses
from generate.answer import generate_answer

FINAL_TOP_K = 2  # chunks given to the LLM (same default eval used so far)


def free_gpu_models() -> None:
    """Forget the cached embedding model + reranker and clear GPU memory."""
    embed_module._model = None
    rerank_module._reranker = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_query(question: str) -> dict:
    """Returns {"answer": str, "chunks": [chunk dicts with source/header_path/text]}."""
    candidates = hybrid_search(question)
    chunks = rerank_module.rerank(question, candidates, top_k=FINAL_TOP_K)
    free_gpu_models()
    result = generate_answer(question, chunks)
    return {"answer": result["answer"], "chunks": chunks}


def format_sources(chunks: list[dict]) -> str:
    """'[1] ddr_condor7_day04.md -- Mud Properties' per chunk, same numbering the LLM saw."""
    return "\n".join(
        f"[{i}] {Path(c['source']).name} -- {c.get('header_path') or '(no section)'}"
        for i, c in enumerate(chunks, start=1)
    )
