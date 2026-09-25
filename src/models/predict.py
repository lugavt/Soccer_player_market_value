"""Load the trained model and produce predictions + SHAP explanations."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.features.engineer import FEATURE_COLS, select_features

MODELS_DIR = Path(__file__).parents[2] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
INTERVALS_PATH = MODELS_DIR / "xgb_intervals.pkl"


def load_interval_model(path=INTERVALS_PATH):
    """The calibrated quantile bundle for src.models.intervals.predict_intervals(), or None."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path=DEFAULT_MODEL_PATH):
    """Return the fitted estimator.

    The pickle is a dict of `{"model": estimator, "features": [...]}` — see
    `train.py`. Unwrap it here so callers get something with `.predict()`.
    Older bare-estimator pickles are still accepted.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"] if isinstance(obj, dict) else obj


def load_feature_names(path=DEFAULT_MODEL_PATH):
    """Feature order the model was actually trained on.

    Should match `FEATURE_COLS`; if it ever drifts, the pickle is authoritative
    and `FEATURE_COLS` needs updating.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["features"] if isinstance(obj, dict) else list(FEATURE_COLS)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Return predicted market values in EUR (back-transformed from log)."""
    return np.exp(model.predict(select_features(X)))


def explain(model, X: pd.DataFrame) -> dict:
    """SHAP explanation for a single player row.

    All values are in log-space. The guarantee
    `baseline_log + sum(shap_values) == pred_log` holds exactly in log-space
    and *not* after back-transforming, so never render raw SHAP values as EUR.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(select_features(X))
    pred_log = float(explainer.expected_value + sv[0].sum())
    return {
        "baseline_log": float(explainer.expected_value),
        "baseline_eur": float(np.exp(explainer.expected_value)),
        "shap_values": dict(zip(FEATURE_COLS, sv[0])),
        "pred_log": pred_log,
        "pred_eur": float(np.exp(pred_log)),
    }
