"""Shared metric functions for categorical and dimensional (VAD) scoring.

Pearson-nan-on-constant-array convention mirrors the sibling PLM repo's
metrics.py: std < 1e-8 on either array -> nan, rather than letting scipy
raise/warn on a constant input.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from src.data import LABELS

_DIMS = ("v", "a", "d")


def accuracy(gold: list[str], pred: list[str]) -> float:
    n = len(gold)
    if n == 0:
        return float("nan")
    return sum(g == p for g, p in zip(gold, pred)) / n


def confusion_matrix(
    gold: list[str], pred: list[str], labels: list[str] = LABELS
) -> list[list[int]]:
    """Rows = gold, cols = pred, ordered by `labels`."""
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        matrix[idx[g]][idx[p]] += 1
    return matrix


def per_class_f1(
    gold: list[str], pred: list[str], labels: list[str] = LABELS
) -> dict[str, float]:
    out = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[label] = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return out


def macro_f1(gold: list[str], pred: list[str], labels: list[str] = LABELS) -> float:
    if not gold:
        return float("nan")
    f1s = per_class_f1(gold, pred, labels)
    return float(np.mean(list(f1s.values())))


def pearson_per_dim(preds: np.ndarray, golds: np.ndarray) -> dict[str, float]:
    """Per V/A/D dim: nan if either array's std < 1e-8, else Pearson r."""
    result = {}
    for i, dim in enumerate(_DIMS):
        p, g = preds[:, i], golds[:, i]
        if np.std(p) < 1e-8 or np.std(g) < 1e-8:
            result[dim] = float("nan")
        else:
            result[dim] = float(pearsonr(p, g)[0])
    return result


def rmse_per_dim(preds: np.ndarray, golds: np.ndarray) -> dict[str, float]:
    return {
        dim: float(np.sqrt(np.mean((preds[:, i] - golds[:, i]) ** 2)))
        for i, dim in enumerate(_DIMS)
    }


def mae_per_dim(preds: np.ndarray, golds: np.ndarray) -> dict[str, float]:
    return {
        dim: float(np.mean(np.abs(preds[:, i] - golds[:, i])))
        for i, dim in enumerate(_DIMS)
    }


def mean_ignore_nan(d: dict[str, float]) -> float:
    vals = [v for v in d.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def granularity_diagnostic(preds: np.ndarray) -> dict[str, int]:
    """Count of distinct predicted values per V/A/D dim.

    Detects integer-collapse (model ignoring the 1-decimal instruction).
    """
    return {dim: int(len(np.unique(preds[:, i]))) for i, dim in enumerate(_DIMS)}
