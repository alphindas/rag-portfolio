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

import re

import ollama

OLLAMA_MODEL = "llama3.2:3b"
# Context window in tokens: it must hold the prompt AND the answer. 2560 is the
# largest size at which llama3.2:3b ran 100% on the GPU on the 4 GB card this was
# built on (Ollama 0.33.1, `ollama ps`); from 2624 up, part of it moved to the CPU.
NUM_CTX = 2560
NUM_PREDICT = 512   # maximum answer length in tokens


class PromptTooLongError(RuntimeError):
    """The prompt does not fit in NUM_CTX, so no answer was generated."""

    def __init__(self, prompt_tokens: int | None, num_ctx: int):
        self.prompt_tokens, self.num_ctx = prompt_tokens, num_ctx
        size = f"{prompt_tokens} tokens" if prompt_tokens else "too long"
        super().__init__(f"Prompt is {size}, but num_ctx is {num_ctx}. No answer was generated, because "
                         "Ollama would have silently cut off part of the context. "
                         "Use fewer or shorter chunks, or raise NUM_CTX.")


def _chat(prompt: str):
    """One Ollama chat call that REFUSES an over-long prompt.

    By default Ollama silently drops part of a prompt that exceeds num_ctx.
    With "truncate": false the server rejects it instead and reports the real
    token count (its own tokenizer, so no estimate on our side, and no extra
    call). ollama-python 0.6.3 chat() has no `truncate` argument, hence the
    private _request(), which keeps OLLAMA_HOST handling and error types.
    Checked with Ollama server 0.33.1; a server that does not know the field
    would ignore it. "shift": false likewise stops the server from dropping old
    tokens if the context fills up while the answer is being written.
    """
    body = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "truncate": False,
        "shift": False,
        "options": {
            "temperature": 0,  # same prompt -> same answer (repeatable evals)
            "seed": 0,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
        "keep_alive": 0,  # unload the model from VRAM immediately after this response
    }
    try:
        return ollama.Client()._request(ollama.ChatResponse, "POST", "/api/chat", json=body)
    except ollama.ResponseError as e:
        if "exceed_context_size" not in str(e.error):
            raise
        count = re.search(r"n_prompt_tokens\W+(\d+)", str(e.error))
        raise PromptTooLongError(int(count.group(1)) if count else None, NUM_CTX) from e


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

    Raises PromptTooLongError instead of answering from a cut-off prompt. The
    result has a "warning" key if the prompt fits but leaves less than
    NUM_PREDICT tokens for the answer.
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

If the context does not contain the answer, reply exactly: "Not documented in the provided context." Do not guess and do not use outside knowledge.

Citation rule: every sentence that uses information from the context MUST end with a citation like [1]. There is no such thing as "no citation needed" -- if a sentence states a fact from the context, it gets a number. If a sentence is pure connecting language with no factual content (e.g. "Here's how:"), it can skip the citation, but any code, step, or claim must be cited.

Example of correct style:
"The widget re-renders when its state changes [1]. Call self.rerender() to trigger this manually [1]."
"""

    response = _chat(user_prompt)

    result = {"answer": response["message"]["content"], "sources": sources_list}
    prompt_tokens = response["prompt_eval_count"]
    if prompt_tokens and prompt_tokens > NUM_CTX - NUM_PREDICT:
        result["warning"] = (f"Prompt is {prompt_tokens} of {NUM_CTX} tokens, leaving fewer than {NUM_PREDICT} "
                             "for the answer, so the answer may be cut off.")
    return result


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