"""An over-long prompt must be reported, never silently cut off. No Ollama needed."""
import json

import ollama
import pytest

import pipeline
from generate import answer as answer_module
from observability import tracer

CHUNKS = [{"text": "Mud weight: 9.2 ppg.", "header_path": "Report", "source": "a.md", "chunk_id": "a::0"}]
# The HTTP body Ollama 0.33.1 returns when "truncate" is false and the prompt
# does not fit: JSON inside a JSON string.
TOO_LONG = json.dumps({"error": json.dumps({"error": {
    "code": 400, "type": "exceed_context_size_error",
    "message": "request (5000 tokens) exceeds the available context size (4096 tokens), try increasing it",
    "n_prompt_tokens": 5000, "n_ctx": 4096}})})


def fake_server(monkeypatch, response=None, error=None) -> dict:
    sent = {}

    def request(self, cls, method, path, json):
        sent.update(json)
        if error:
            raise ollama.ResponseError(error, 400)
        return response

    monkeypatch.setattr(ollama.Client, "_request", request)
    return sent


def test_request_forbids_truncation_and_uses_num_ctx(monkeypatch):
    sent = fake_server(monkeypatch, {"message": {"content": "9.2 ppg [1]."}, "prompt_eval_count": 700})
    result = answer_module.generate_answer("Mud weight?", CHUNKS)
    assert sent["truncate"] is False and sent["shift"] is False
    assert sent["options"]["num_ctx"] == answer_module.NUM_CTX
    assert "warning" not in result


def test_too_long_prompt_raises_with_token_counts(monkeypatch):
    fake_server(monkeypatch, error=TOO_LONG)
    with pytest.raises(answer_module.PromptTooLongError, match=f"5000 tokens, but num_ctx is {answer_module.NUM_CTX}"):
        answer_module.generate_answer("Mud weight?", CHUNKS)


def test_other_ollama_errors_are_not_swallowed(monkeypatch):
    fake_server(monkeypatch, error="CUDA error: device busy")
    with pytest.raises(ollama.ResponseError):
        answer_module.generate_answer("Mud weight?", CHUNKS)


def test_warns_when_little_room_is_left_for_the_answer(monkeypatch):
    nearly_full = answer_module.NUM_CTX - answer_module.NUM_PREDICT + 1
    fake_server(monkeypatch, {"message": {"content": "x [1]"}, "prompt_eval_count": nearly_full})
    assert "may be cut off" in answer_module.generate_answer("Mud weight?", CHUNKS)["warning"]


def test_pipeline_returns_the_warning_and_traces_it(monkeypatch, tmp_path):
    fake_server(monkeypatch, error=TOO_LONG)
    monkeypatch.setattr(tracer, "TRACE_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr(pipeline, "hybrid_search", lambda question, trace: CHUNKS)
    monkeypatch.setattr(pipeline, "free_gpu_models", lambda: None)
    result = pipeline.run_query("Mud weight?", use_reranker=False)
    assert result["answer"] == "" and result["citations"] is None
    assert "5000 tokens" in result["warning"]
    assert "5000 tokens" in json.loads((tmp_path / "traces.jsonl").read_text())["warning"]
