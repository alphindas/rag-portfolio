"""Subset selection, resume bookkeeping and rate-limit parsing. No API calls."""
import pytest

from eval import scores

QUESTIONS = [{"id": f"q{i:02d}"} for i in range(1, 46)]


def test_subset_is_fixed_and_ordered():
    ids = [q["id"] for q in scores.select_subset(QUESTIONS, 30, 42)]
    assert ids == [q["id"] for q in scores.select_subset(QUESTIONS, 30, 42)]
    assert len(set(ids)) == 30 and ids == sorted(ids)
    assert scores.select_subset(QUESTIONS, None) == QUESTIONS


def test_pending_scores_one_metric_for_all_runs_first_and_resumes(tmp_path):
    path = str(tmp_path / "s.json")
    store = scores.load_store(["q01", "q02"], 42, path)
    todo = scores.pending(store, ["a", "b"])
    assert [t[:2] for t in todo[:5]] == [("faithfulness", "a")] * 2 + [("faithfulness", "b")] * 2 + [("answer_relevancy", "a")]
    store["scores"] = {"a": {"faithfulness": {"q01": 1.0, "q02": None}}}  # null = tried, not repeated
    scores.save_store(store, path)
    assert scores.pending(scores.load_store(["q01", "q02"], 42, path), ["a", "b"]) == todo[2:]


def test_metric_selection_keeps_scores_apart():
    assert scores.parse_metrics(None) == scores.METRIC_NAMES
    assert scores.parse_metrics("answer_relevancy, faithfulness,faithfulness") == ["faithfulness", "answer_relevancy"]
    for bad in ("faithfullness", "", "faithfulness,nope"):
        with pytest.raises(SystemExit):
            scores.parse_metrics(bad)
    store = {"question_ids": ["q01"], "scores": {"a": {"faithfulness": {"q01": 0.5}}}}
    assert scores.pending(store, ["a"], ["faithfulness"]) == []
    assert scores.pending(store, ["a"], ["answer_relevancy"]) == [("answer_relevancy", "a", "q01")]
    assert len(scores.pending(store, ["a"])) == 3  # default: everything not yet scored


def test_store_refuses_to_mix_judges_or_question_sets(tmp_path):
    path = str(tmp_path / "s.json")
    store = scores.load_store(["q01"], 42, path)
    scores.save_store(store, path)
    with pytest.raises(SystemExit):
        scores.load_store(["q02"], 42, path)
    scores.save_store({**store, "judge_model": "other"}, path)
    with pytest.raises(SystemExit):
        scores.load_store(["q01"], 42, path)


def test_rate_limit_detection():
    minute = Exception("Error code: 429 - rate_limit_exceeded on tokens per minute (TPM). Please try again in 7.5s.")
    daily = Exception("Error code: 429 - rate_limit_exceeded on tokens per day (TPD). Please try again in 1m2.5s.")
    assert scores.is_rate_limit(minute) and not scores.is_daily_limit(minute)
    assert scores.is_daily_limit(daily)
    assert scores.retry_wait_seconds(minute) == 8.5
    assert scores.retry_wait_seconds(daily) == 63.5
    assert scores.retry_wait_seconds(Exception("try again in 500ms")) == 1.5
    assert scores.retry_wait_seconds(Exception("no hint")) == 60.0
    assert not scores.is_rate_limit(ValueError("bad JSON"))
