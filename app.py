"""
app.py -- Streamlit demo.   streamlit run app.py

Type a question, see the answer, the cited chunks and per-stage timing.
The sidebar toggle turns the reranker on/off so you can see its effect live.

GPU note (4 GB card): the embedder and reranker are loaded ONCE and kept in
RAM; they visit the GPU only during retrieval, then step aside so Ollama can
load the LLM. Ollama must be running on the same machine.
"""
import re

import streamlit as st

from pipeline import FINAL_TOP_K, format_sources, run_query, warm_up

st.set_page_config(page_title="RAG over drilling reports", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model and reranker (once)...")
def load_models() -> float:
    return warm_up()


load_seconds = load_models()

st.title("Ask the drilling reports")
st.caption("Synthetic daily drilling reports. Hybrid search (BM25 + vectors) -> optional reranker -> local LLM with [n] citations.")

with st.sidebar:
    use_reranker = st.toggle("Use reranker", value=True,
                             help="Off: the top chunks come straight from hybrid search. On: a cross-encoder re-orders the top 10.")
    st.caption(f"Chunks given to the LLM: {FINAL_TOP_K}")
    st.caption(f"Models loaded at startup in {load_seconds:.1f} s (only paid once).")

question = st.text_input("Question", placeholder="What was the mud weight on Condor-9 on Day 5?")

if question:
    try:
        with st.spinner("Searching and generating..."):
            result = run_query(question, source="app", use_reranker=use_reranker, keep_models=True)
    except ConnectionError:
        st.error("Could not reach Ollama. Start it with `ollama serve`.")
        st.stop()

    st.subheader("Answer")
    st.write(result["answer"])
    status = result["citations"]["status"]
    if status == "no_citations":
        st.warning("This answer has no [n] citations, so it is not verified against the sources.")
    elif status == "invalid":
        st.warning(f"This answer cites sources that do not exist: {result['citations']['invalid']}")

    st.subheader("Timing")
    timings = result["timings"]
    cols = st.columns(len(timings) + 1)
    for col, (name, ms) in zip(cols, timings.items()):
        col.metric(name, f"{ms:.0f} ms")
    cols[-1].metric("total", f"{sum(timings.values()):.0f} ms")

    st.subheader("Sources")
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    for i, chunk in enumerate(result["chunks"], start=1):
        label = format_sources([chunk]).replace("[1]", f"[{i}]", 1)
        with st.expander(label + ("  (cited)" if i in cited else ""), expanded=i in cited):
            st.text(chunk["text"])
