# eval/ragas_judge.py
import os
import sys
import types
from dotenv import load_dotenv

load_dotenv()

# --- Workaround for a known ragas bug (explodinggradients/ragas#2753) ---
# ragas/llms/base.py unconditionally imports ChatVertexAI from a path that
# langchain-community removed. We never use Vertex AI, so we register a
# dummy module here to satisfy that import before ragas runs it.
def _stub_missing_import(module_path: str, symbol: str) -> None:
    if module_path in sys.modules:
        return
    stub = types.ModuleType(module_path)
    setattr(stub, symbol, type(symbol, (), {}))
    sys.modules[module_path] = stub

_stub_missing_import("langchain_community.chat_models.vertexai", "ChatVertexAI")
# --- end workaround ---

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def get_evaluator_llm():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to .env before running eval."
        )
    chat_model = ChatGroq(
        # Temporarily on openai/gpt-oss-20b: openai/gpt-oss-120b's free-tier
        # daily quota (200K tokens/day) got exhausted during debugging today.
        # llama-3.1-8b-instant (tried first) turned out to be deprecated by
        # Groq as of June 17, 2026 -- gpt-oss-20b is Groq's own recommended
        # replacement for it, with a separate quota untouched by today's
        # debugging. Weaker judge than the 120b model -- swap back once its
        # quota resets, for a stronger judge.
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=4096,  # faithfulness scoring asks the judge to list out
                           # every factual statement in an answer before
                           # checking each one -- without an explicit ceiling,
                           # ChatGroq falls back to a default too small for
                           # that, truncating mid-response on longer answers
        api_key=groq_key,
    )
    return LangchainLLMWrapper(chat_model)


def get_evaluator_embeddings():
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    return LangchainEmbeddingsWrapper(hf_embeddings)