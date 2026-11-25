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
import time
from pathlib import Path

from retrieve.hybrid import hybrid_search  # also puts ingest/ on sys.path
import retrieve.rerank as rerank_module
import embed as embed_module  # the same module object hybrid.py uses
from generate.answer import PromptTooLongError, generate_answer
from generate.citations import check_citations
from observability.tracer import Trace

FINAL_TOP_K = 2  # chunks given to the LLM (same default eval used so far)


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


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


def _move_models(device: str) -> None:
    for model in (embed_module._model, rerank_module._reranker):
        if model is not None:
            model.to(device)


def park_models() -> None:
    """Keep the models loaded but move them from GPU to system RAM (the GPU
    then holds only ~100 MiB of CUDA context), so the LLM fits."""
    _move_models("cpu")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def warm_up() -> float:
    """Load the embedder and reranker once (the slow disk read) and park them.
    Returns the seconds it took. Used by the app at startup."""
    t0 = time.perf_counter()
    embed_module._get_model()
    rerank_module._get_reranker()
    park_models()
    return time.perf_counter() - t0


def run_query(question: str, source: str = "cli", use_reranker: bool = True, keep_models: bool = False) -> dict:
    """
    Returns {"answer", "chunks", "citations" (checked in code), "warning", "timings" (ms per stage)}.
    If the prompt is too long for the LLM, "answer" is "", "citations" is None
    and "warning" says why.
    Also appends one line describing the query to logs/traces.jsonl.

    keep_models=False (CLI): drop the embedder + reranker after retrieval, so
        every question pays to load them again.
    keep_models=True (app): keep them in RAM between questions and only move
        them to the GPU for retrieval.
    use_reranker=False: skip the cross-encoder, use the fused order directly.
    """
    trace = Trace(question, source=source, reranker=use_reranker)
    if keep_models:
        _move_models("cuda" if _cuda() else "cpu")
    candidates = hybrid_search(question, trace=trace)
    if use_reranker:
        chunks = rerank_module.rerank(question, candidates, top_k=FINAL_TOP_K, trace=trace)
    else:
        chunks = candidates[:FINAL_TOP_K]
    park_models() if keep_models else free_gpu_models()
    try:
        with trace.stage("generate"):
            result = generate_answer(question, chunks)
    except PromptTooLongError as e:
        result = {"answer": None, "warning": str(e)}
    answer, warning = result["answer"], result.get("warning")
    citations = None if answer is None else check_citations(answer, len(chunks))
    trace.finish(answer=answer, citations=citations, **({"warning": warning} if warning else {}))
    return {"answer": answer or "", "chunks": chunks, "citations": citations, "warning": warning,
            "timings": trace.timings()}


def format_sources(chunks: list[dict]) -> str:
    """'[1] ddr_condor7_day04.md -- Mud Properties' per chunk, same numbering the LLM saw."""
    return "\n".join(
        f"[{i}] {Path(c['source']).name} -- {c.get('header_path') or '(no section)'}"
        for i, c in enumerate(chunks, start=1)
    )
