# Streamlit demo in a container.
#
# IMPORTANT: Ollama (the LLM) is NOT in this image. It runs on the HOST, and the
# container reaches it over the network (see the README "Docker" section).
#
#   docker build -t rag-portfolio .
#   docker run --rm -p 8501:8501 --add-host=host.docker.internal:host-gateway rag-portfolio
#
# This image uses CPU-only PyTorch (small, works anywhere): the embedder and
# reranker run on the CPU inside the container. The LLM still uses your GPU,
# because Ollama runs on the host.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf

# CPU torch first, so the pinned `torch==...` in requirements.txt is already satisfied.
COPY requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# Download the models and build the index at build time, so the first start is not slow.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en'); CrossEncoder('BAAI/bge-reranker-base')" \
    && python ingest/build_index.py

# Where Ollama lives. host.docker.internal = the machine running Docker.
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
