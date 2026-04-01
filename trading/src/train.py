"""
train.py — XGBoost model training and walk-forward loop.

The walk-forward loop:
  1. For each (train_idx, val_idx) from the CV splitter:
     a. Slice X, y by position.
     b. Fit XGBoost on X_train, y_train.
     c. Predict on X_val.
     d. Collect (timestamp, y_true, y_pred) for each val bar.
  2. Concatenate all out-of-sample predictions.
  3. Optionally save per-fold models to disk.

No future data touches the training set at any point.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .feature_engineering import get_feature_names
from .utils import ensure_dir, get_logger, resolve_path

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_cfg: dict) -> xgb.XGBRegressor:
    """
    Instantiate an XGBRegressor from the 'model' config section.

    All hyperparameters are explicitly sourced from config — no hidden defaults.
    """
    params = {
        "n_estimators":     model_cfg.get("n_estimators", 500),
        "max_depth":        model_cfg.get("max_depth", 4),
        "learning_rate":    model_cfg.get("learning_rate", 0.04),
        "subsample":        model_cfg.get("subsample", 0.8),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "min_child_weight": model_cfg.get("min_child_weight", 5),
        "reg_alpha":        model_cfg.get("reg_alpha", 0.1),
        "reg_lambda":       model_cfg.get("reg_lambda", 1.0),
        "objective":        model_cfg.get("objective", "reg:squarederror"),
        "eval_metric":      model_cfg.get("eval_metric", "rmse"),
        "random_state":     model_cfg.get("random_state", 42),
        "n_jobs":           model_cfg.get("n_jobs", -1),
        "verbosity":        0,
    }
    return xgb.XGBRegressor(**params)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_walk_forward(
    model_df: pd.DataFrame,
    cv,
    model_cfg: dict,
    output_cfg: dict,
) -> pd.DataFrame:
    """
    Run the full walk-forward validation loop.

    Parameters
    ----------
    model_df : pd.DataFrame
        Feature matrix with a 'target' column, indexed by timestamp.
    cv : ExpandingWindowCV or RollingWindowCV
        The CV splitter.
    model_cfg : dict
        The 'model' section of config.yaml.
    output_cfg : dict
        The 'output' section of config.yaml.

    Returns
    -------
    pd.DataFrame
        All out-of-sample predictions concatenated, with columns:
        ['y_true', 'y_pred', 'fold'].
        Indexed by the original timestamps.
    """
    feature_names = get_feature_names(model_df)
    X = model_df[feature_names]
    y = model_df["target"]

    save_models = output_cfg.get("save_fold_models", True)
    artifacts_dir = ensure_dir(output_cfg["artifacts_dir"])
    early_stop = model_cfg.get("early_stopping_rounds", None)

    all_predictions: list[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(model_df)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = build_model(model_cfg)

        # Fit — use early stopping if configured and we have enough train data
        if early_stop is not None:
            # Use last 10% of training set as internal eval set for early stopping.
            # This is fine: it's still strictly in the past relative to val_idx.
            split_at = max(1, int(len(X_train) * 0.9))
            eval_X = X_train.iloc[split_at:]
            eval_y = y_train.iloc[split_at:]
            X_train_fit = X_train.iloc[:split_at]
            y_train_fit = y_train.iloc[:split_at]

            model.set_params(early_stopping_rounds=early_stop)
            model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(eval_X, eval_y)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_val)

        fold_preds = pd.DataFrame(
            {"y_true": y_val.values, "y_pred": y_pred, "fold": fold_idx},
            index=y_val.index,
        )
        all_predictions.append(fold_preds)

        # Per-fold quick stats
        corr = np.corrcoef(y_val.values, y_pred)[0, 1]
        sign_acc = np.mean(np.sign(y_pred) == np.sign(y_val.values))
        logger.info(
            "Fold %d | val_bars=%d | corr=%.4f | sign_acc=%.4f",
            fold_idx, len(val_idx), corr, sign_acc,
        )

        # Save fold model
        if save_models:
            model_path = artifacts_dir / f"model_fold_{fold_idx}.pkl"
            with model_path.open("wb") as fh:
                pickle.dump(model, fh)

    # Concatenate all out-of-sample predictions
    oos = pd.concat(all_predictions)

    # Save feature names for later use (SHAP, inspection)
    feat_path = artifacts_dir / "feature_names.json"
    with feat_path.open("w") as fh:
        json.dump(feature_names, fh)
    logger.info("Saved feature names → %s", feat_path)

    # Save the last fold's model as the "final" reference model
    last_model_path = artifacts_dir / "model_last_fold.pkl"
    with last_model_path.open("wb") as fh:
        pickle.dump(model, fh)  # 'model' is still the last fold's model
    logger.info("Saved last-fold model → %s", last_model_path)

    # Persist OOS predictions
    preds_path = resolve_path(output_cfg["predictions_path"])
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    oos.to_parquet(preds_path)
    logger.info("Saved OOS predictions → %s (%d rows)", preds_path, len(oos))

    return oos
