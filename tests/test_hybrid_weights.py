import pandas as pd
import pytest

from str_suitability.suitability.hybrid_weights import (
    bloc_weight,
    equal_weights,
    hybrid_weights,
    rf_importance_weights,
)


def rows(**values) -> list[dict]:
    return [{"feature": name, "importance_mean": value} for name, value in values.items()]


def test_equal_weights_are_uniform_and_sum_to_one():
    weights = equal_weights(["a", "b", "c", "d"])
    assert weights.tolist() == [0.25, 0.25, 0.25, 0.25]
    assert weights.sum() == pytest.approx(1.0)


def test_rf_importance_clips_negative_importances_to_zero():
    weights = rf_importance_weights([rows(a=2.0, b=-1.0)], ["a", "b"])
    assert weights["a"] == pytest.approx(1.0)
    assert weights["b"] == pytest.approx(0.0)


def test_rf_importance_reindexes_to_the_indicator_set():
    weights = rf_importance_weights([rows(a=1.0, extra=9.0)], ["a", "b"])
    assert list(weights.index) == ["a", "b"]
    assert weights["a"] == pytest.approx(1.0)
    assert weights["b"] == pytest.approx(0.0)


def test_rf_importance_drops_a_model_whose_importances_are_all_negative():
    weights = rf_importance_weights([rows(a=1.0, b=1.0), rows(a=-1.0, b=-1.0)], ["a", "b"])
    assert weights["a"] == pytest.approx(0.5)
    assert weights["b"] == pytest.approx(0.5)


def test_rf_importance_is_zero_when_no_model_reports_positive_importance():
    weights = rf_importance_weights([rows(a=-1.0, b=-2.0)], ["a", "b"])
    assert (weights == 0.0).all()


def test_hybrid_weights_blend_and_sum_to_one():
    entropy = pd.Series({"x": 0.9, "y": 0.1})
    rfi = pd.Series({"x": 0.1, "y": 0.9})
    blended = hybrid_weights(entropy, rfi, blend_ratio=0.5)
    assert blended.sum() == pytest.approx(1.0)
    assert blended["x"] == pytest.approx(0.5)
    assert blended["y"] == pytest.approx(0.5)


def test_hybrid_weights_without_rfi_keep_the_entropy_weight():
    entropy = pd.Series({"x": 0.7, "y": 0.3})
    rfi = pd.Series({"x": 1.0, "y": float("nan")})
    blended = hybrid_weights(entropy, rfi, blend_ratio=0.5)
    assert blended["y"] == pytest.approx(0.3 / 1.15)
    assert blended.sum() == pytest.approx(1.0)


def test_blend_ratio_one_recovers_the_entropy_weights():
    entropy = pd.Series({"x": 0.8, "y": 0.2})
    rfi = pd.Series({"x": 0.0, "y": 1.0})
    assert hybrid_weights(entropy, rfi, blend_ratio=1.0).tolist() == pytest.approx(
        entropy.tolist()
    )


def test_bloc_weight_sums_only_present_members():
    weights = pd.Series({"a": 0.5, "b": 0.3, "c": 0.2})
    assert bloc_weight(weights, ["a", "b", "missing"]) == pytest.approx(0.8)
