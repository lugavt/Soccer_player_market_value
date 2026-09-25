"""Calibrated prediction intervals: conformalized quantile regression.

A second XGBoost model predicts the 10/25/50/75/90% quantiles of log(value).
On its own it is overconfident on unseen seasons — its "80%" interval held the
listed value only ~60% of the time on the test seasons. So each interval is
widened by the error observed on the validation season (split-conformal
calibration, "CQR"): after that the 80% interval covers ~79% of test players.

Intervals are in log-space internally and exponentiated at the end, so they are
asymmetric in euros — right for a lognormal quantity like market value.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.engineer import select_features

QUANTILES = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
LEVELS = {0.8: (0.10, 0.90), 0.5: (0.25, 0.75)}   # interval level -> (lower q, upper q)


def train_quantile_model(X_train, y_train, X_val, y_val, params: dict) -> dict:
    """Fit the multi-quantile model and calibrate each interval on the validation season.

    Returns the bundle predict_intervals() needs: model, quantiles and one
    conformal margin (log-space) per interval level. The validation season also
    drives early stopping, which makes the calibration very slightly optimistic;
    test-season coverage is reported by train.py so that stays visible.
    """
    model = xgb.XGBRegressor(**{**params, "objective": "reg:quantileerror",
                                "quantile_alpha": QUANTILES, "eval_metric": "quantile"})
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    q = np.sort(model.predict(X_val), axis=1)        # sorting removes rare quantile crossing
    y = np.asarray(y_val)
    margins = {}
    for level, (lo_q, hi_q) in LEVELS.items():
        lo, hi = q[:, _col(lo_q)], q[:, _col(hi_q)]
        scores = np.maximum(lo - y, y - hi)          # how far outside the interval each point fell
        rank = min(1.0, level * (1 + 1 / len(y)))    # finite-sample conformal correction
        margins[level] = float(np.quantile(scores, rank))
    return {"model": model, "quantiles": QUANTILES.tolist(), "margins": margins}


def _col(q: float) -> int:
    return int(np.argmin(np.abs(QUANTILES - q)))


def predict_intervals(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """low_80 / low_50 / high_50 / high_80 in EUR for each row of X."""
    q = np.sort(bundle["model"].predict(select_features(X)), axis=1)
    out = {}
    for level, (lo_q, hi_q) in LEVELS.items():
        m = bundle["margins"][level]
        tag = int(level * 100)
        out[f"low_{tag}"] = np.exp(q[:, _col(lo_q)] - m)
        out[f"high_{tag}"] = np.exp(q[:, _col(hi_q)] + m)
    return pd.DataFrame(out, index=X.index)


def coverage(intervals: pd.DataFrame, actual: pd.Series) -> dict:
    """Share of rows whose listed value falls inside each interval."""
    return {f"{tag}%": float(((actual >= intervals[f"low_{tag}"]) & (actual <= intervals[f"high_{tag}"])).mean())
            for tag in (50, 80)}
