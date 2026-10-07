"""
generate/answer.py

Generates an answer to a query using the top reranked chunks as context,
via a local Ollama model. Citations are enforced through prompting
(soft enforcement) -- the model is instructed to cite [n] per claim,
mapped to a numbered sources list built from the reranked chunks.

Why local Ollama instead of a hosted API?
------------------------------------------------------
Keeps the whole pipeline runnable offline with no API costs -- useful
for a portfolio project you want to demo without burning API credits
or needing keys. Same reason the eval judge (RAGAS) points at Ollama.
Trade-off, stated plainly: a local 8B model is meaningfully weaker at
instruction-following than GPT-4/Claude-class models, so citation
discipline will be less reliable than with a hosted model -- that's
a known limitation of this setup, not a hidden one.

Why "soft" citation enforcement (prompting only, no validation)?
------------------------------------------------------
The alternative is code that parses the model's output, checks every
sentence/claim has a [n] marker, and rejects/retries if not. That's a
reasonable thing to build LATER (it pairs naturally with the eval
step), but it adds real complexity: sentence segmentation, deciding
what counts as a "claim" needing a citation, a retry loop. For now we
rely on the prompt instruction alone and treat citation coverage as
something eval/ will measure and report on -- not something generation
silently guarantees. This is worth revisiting once eval/ exists and we
can see empirically how often the local model skips citations.

Citation format
------------------------------------------------------
The prompt asks the model to cite claims inline as [1], [2], etc.,
where the numbers correspond to the order of chunks in the context
(not to chunk_id or rerank score). We build a numbered sources list
after generation so the reader can map [1] -> which chunk in the
sample_docs it points to.
"""

from __future__ import annotations

import ollama

OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context chunks.

Rules:
- Every factual claim in your answer MUST be followed by a citation marker like [1], [2], etc., referring to the numbered context chunk it came from.
- If a claim draws on multiple chunks, cite all of them, e.g. [1][2].
- If the context does not contain enough information to answer, say so plainly instead of guessing.
- Do not invent information that isn't in the provided context.
- Keep the answer concise and directly responsive to the question."""


def _build_context_block(chunks: list[dict]) -> str:
    """
    Formats reranked chunks into a numbered context block for the
    prompt, e.g.:
        [1] (Widget Framework > Installation)
        <chunk text>

        [2] (Widget Framework > Usage)
        <chunk text>
    The numbering here is what the model's [1]/[2] citations refer to --
    it's positional (order in this list), not the chunk_id.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        header = chunk.get("header_path") or "(no header)"
        lines.append(f"[{i}] ({header})\n{chunk['text']}")
    return "\n\n".join(lines)


def _build_sources_list(chunks: list[dict]) -> str:
    """
    Formats the same numbering into a human-readable sources list to
    append after the answer, so [1]/[2] in the answer text resolves to
    something concrete (source file + header_path).
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        header = chunk.get("header_path") or "(no header)"
        source = chunk.get("source", "unknown source")
        lines.append(f"[{i}] {header} -- {source}")
    return "\n".join(lines)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Calls local Ollama with the query + numbered context chunks, and
    returns a dict with the raw answer text and a formatted sources list.

    chunks: the reranked list from retrieve/rerank.py (top_k results,
    each with at least "text", "header_path", "source").
    """
    if not chunks:
        return {
            "answer": "No relevant context was found to answer this question.",
            "sources": "",
        }

    context_block = _build_context_block(chunks)
    sources_list = _build_sources_list(chunks)

    user_prompt = f"""Context:
{context_block}

Question: {query}

Answer the question using only the context above.

Citation rule: every sentence that uses information from the context MUST end with a citation like [1]. There is no such thing as "no citation needed" -- if a sentence states a fact from the context, it gets a number. If a sentence is pure connecting language with no factual content (e.g. "Here's how:"), it can skip the citation, but any code, step, or claim must be cited.

Example of correct style:
"The widget re-renders when its state changes [1]. Call self.rerender() to trigger this manually [1]."
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        options={
            "num_predict": 512,
            "num_ctx": 1024,  # reduced from 2048 -- our prompts (5 short DDR
                               # chunks + question) are nowhere near that size,
                               # and a smaller KV-cache needs less contiguous
                               # VRAM to allocate, giving more headroom against
                               # OOMs on this 4GB card
        },
        keep_alive=0,  # unload the model from VRAM immediately after this response
    )

    answer_text = response["message"]["content"]

    return {
        "answer": answer_text,
        "sources": sources_list,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "retrieve"))
    sys.path.insert(0, str(Path(__file__).parent.parent / "ingest"))
    from hybrid import hybrid_search  # local import: retrieve/hybrid.py
    from rerank import rerank  # local import: retrieve/rerank.py

    if len(sys.argv) != 2:
        print("Usage: python answer.py '<query text>'")
        sys.exit(1)

    query = sys.argv[1]

    fused = hybrid_search(query)
    reranked = rerank(query, fused, top_k=5)

    # Free embed + rerank models from GPU before loading the LLM --
    # 4GB VRAM can't hold all three at once.
    import gc
    import torch
    import embed as embed_module
    import rerank as rerank_module
    embed_module._model = None
    rerank_module._reranker = None
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[answer] Generating with {len(reranked)} reranked chunks as context...\n")

    result = generate_answer(query, reranked)

    print("=== Answer ===")
    print(result["answer"])
    print("\n=== Sources ===")
    print(result["sources"])