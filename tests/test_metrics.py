import pytest

from src.evaluation.metrics import mean_metrics, retrieval_metrics_at_k


def test_all_gold_found():
    m = retrieval_metrics_at_k(["a", "b", "c"], {"a", "b"}, k=3)
    assert m["recall"] == 1.0 and m["all_hit"] == 1.0 and m["any_hit"] == 1.0
    assert m["precision"] == pytest.approx(2 / 3)


def test_partial():
    m = retrieval_metrics_at_k(["a", "x", "y"], {"a", "b"}, k=3)
    assert m["recall"] == 0.5 and m["all_hit"] == 0.0 and m["any_hit"] == 1.0


def test_cutoff_respected():
    assert retrieval_metrics_at_k(["x", "a"], {"a"}, k=1)["any_hit"] == 0.0


def test_empty_gold_raises():
    with pytest.raises(ValueError):
        retrieval_metrics_at_k(["a"], set(), 1)


def test_mean():
    assert mean_metrics([{"r": 1.0}, {"r": 0.0}]) == {"r": 0.5}