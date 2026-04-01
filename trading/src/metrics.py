"""
metrics.py — regression and signal quality metrics for OOS predictions.

All metrics are computed on OUT-OF-SAMPLE predictions only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import get_logger, resolve_path

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation between predictions and actuals."""
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of predictions where sign(y_pred) == sign(y_true).

    Rows where y_true == 0 are excluded (ambiguous ground truth).
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "corr": pearson_corr(y_true, y_pred),
        "sign_accuracy": sign_accuracy(y_true, y_pred),
        "n_obs": int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold and aggregate reporting
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(oos: pd.DataFrame, metrics_path: Optional[str] = None) -> dict:
    """
    Compute per-fold metrics and overall metrics from OOS predictions.

    Parameters
    ----------
    oos : pd.DataFrame
        Must have columns ['y_true', 'y_pred', 'fold'].
    metrics_path : str, optional
        If provided, save the full metrics dict as JSON.

    Returns
    -------
    dict with keys 'overall' and 'per_fold'.
    """
    results: dict = {"overall": {}, "per_fold": {}}

    # Overall metrics across all OOS predictions
    overall = compute_metrics(oos["y_true"].values, oos["y_pred"].values)
    results["overall"] = overall

    logger.info(
        "Overall OOS | RMSE=%.6f | MAE=%.6f | Corr=%.4f | SignAcc=%.4f | N=%d",
        overall["rmse"], overall["mae"], overall["corr"],
        overall["sign_accuracy"], overall["n_obs"],
    )

    # Per-fold breakdown
    for fold_id, group in oos.groupby("fold"):
        fold_m = compute_metrics(group["y_true"].values, group["y_pred"].values)
        results["per_fold"][int(fold_id)] = fold_m
        logger.info(
            "  Fold %d | RMSE=%.6f | Corr=%.4f | SignAcc=%.4f | N=%d",
            fold_id, fold_m["rmse"], fold_m["corr"],
            fold_m["sign_accuracy"], fold_m["n_obs"],
        )

    if metrics_path:
        p = resolve_path(metrics_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Saved metrics → %s", p)

    return results
