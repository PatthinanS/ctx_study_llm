import math

import numpy as np
import pytest

from src.metrics import (
    accuracy,
    confusion_matrix,
    granularity_diagnostic,
    macro_f1,
    mae_per_dim,
    mean_ignore_nan,
    pearson_per_dim,
    per_class_f1,
    rmse_per_dim,
)

LABELS = ["ang", "hap", "exc", "neu", "sad", "fru"]
GOLD = ["ang", "hap", "ang", "neu", "hap", "neu"]
PRED = ["ang", "hap", "neu", "neu", "hap", "ang"]


def test_accuracy_hand_computed():
    assert accuracy(GOLD, PRED) == pytest.approx(4 / 6)


def test_per_class_f1_hand_computed():
    f1s = per_class_f1(GOLD, PRED, LABELS)
    expected = {"ang": 0.5, "hap": 1.0, "exc": 0.0, "neu": 0.5, "sad": 0.0, "fru": 0.0}
    for label, val in expected.items():
        assert f1s[label] == pytest.approx(val)


def test_macro_f1_hand_computed():
    assert macro_f1(GOLD, PRED, LABELS) == pytest.approx(2.0 / 6)


def test_confusion_matrix_shape_and_counts():
    m = confusion_matrix(GOLD, PRED, LABELS)
    expected = [
        [1, 0, 0, 1, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    assert m == expected


PREDS_VAD = np.array([[1, 5, 1], [2, 5, 2], [3, 5, 3], [4, 5, 4]], dtype=float)
GOLDS_VAD = np.array([[1, 1, 4], [2, 1, 3], [3, 1, 2], [4, 1, 1]], dtype=float)


def test_pearson_per_dim_known_values():
    r = pearson_per_dim(PREDS_VAD, GOLDS_VAD)
    assert r["v"] == pytest.approx(1.0)
    assert r["d"] == pytest.approx(-1.0)


def test_pearson_per_dim_constant_array_returns_nan():
    r = pearson_per_dim(PREDS_VAD, GOLDS_VAD)
    assert math.isnan(r["a"])  # both preds and golds constant on dim 'a'


def test_rmse_mae_hand_computed():
    rmse = rmse_per_dim(PREDS_VAD, GOLDS_VAD)
    mae = mae_per_dim(PREDS_VAD, GOLDS_VAD)
    assert rmse["v"] == pytest.approx(0.0)
    assert rmse["a"] == pytest.approx(4.0)
    assert rmse["d"] == pytest.approx(math.sqrt(5))
    assert mae["v"] == pytest.approx(0.0)
    assert mae["a"] == pytest.approx(4.0)
    assert mae["d"] == pytest.approx(2.0)


def test_granularity_diagnostic_counts_distinct():
    g = granularity_diagnostic(PREDS_VAD)
    assert g == {"v": 4, "a": 1, "d": 4}


def test_mean_ignore_nan():
    assert mean_ignore_nan({"v": 1.0, "a": float("nan"), "d": -1.0}) == pytest.approx(0.0)
    assert math.isnan(mean_ignore_nan({"v": float("nan")}))
