"""Train the market-value model end to end.

    python -m src.models.train

Reads `data/panel_final.parquet`, writes `models/xgb_model.pkl` and
`data/test_predictions.parquet`. Hyperparameters and the temporal split match
notebook 03 — the notebook is for exploration, this is the reproducible path.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import DATA_DIR, load_panel
from src.features.engineer import FEATURE_COLS, build_features, temporal_split
from src.models.intervals import coverage, predict_intervals, train_quantile_model

MODELS_DIR = Path(__file__).parents[2] / "models"

XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=30,
    eval_metric="rmse",
    verbosity=0,
)

BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"


def model_params() -> dict:
    """Hyperparameters to train with: Optuna's (src/models/tune.py) if present, else XGB_PARAMS.

    Tuned params come with more, slower trees, so early stopping gets more room;
    the fixed settings (seed, metric, verbosity) always come from here.
    """
    if not BEST_PARAMS_PATH.exists():
        return dict(XGB_PARAMS)
    tuned = json.loads(BEST_PARAMS_PATH.read_text())["params"]
    return {**XGB_PARAMS, "n_estimators": 3000, "early_stopping_rounds": 50, **tuned}


def evaluate(name, model, X, y) -> dict:
    """Metrics in both log-space (what we fit) and EUR (what clubs read).

    MAPE is a *median* absolute percentage error — the value distribution is
    lognormal, so a mean would be dragged around by a handful of superstars.
    """
    pred_log = model.predict(X)
    metrics = {
        "r2": r2_score(y, pred_log),
        "rmse_log": float(np.sqrt(np.mean((y - pred_log) ** 2))),
        "mae_log": mean_absolute_error(y, pred_log),
        "mape_eur": float(np.median(np.abs((np.exp(y) - np.exp(pred_log)) / np.exp(y))) * 100),
    }
    print(
        f"  {name:<8} R²={metrics['r2']:.3f}  RMSE(log)={metrics['rmse_log']:.3f}  "
        f"MAE(log)={metrics['mae_log']:.3f}  MAPE(€)={metrics['mape_eur']:.1f}%"
    )
    return metrics


def train_baseline(X_train, y_train, splits):
    """Ridge on median-imputed, standardised features — the bar XGBoost must clear."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10)),
    ])
    pipe.fit(X_train, y_train)
    print("=== RIDGE BASELINE ===")
    for name, (X, y, _) in splits.items():
        evaluate(name, pipe, X, y)
    return pipe


def train_xgb(X_train, y_train, X_val, y_val, splits):
    params = model_params()
    source = "tuned (models/best_params.json)" if BEST_PARAMS_PATH.exists() else "hand-set XGB_PARAMS"
    print(f"hyperparameters: {source}")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("=== XGBOOST ===")
    for name, (X, y, _) in splits.items():
        evaluate(name, model, X, y)
    return model


def main(skip_baseline: bool = False):
    df = build_features(load_panel())
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, test_rows) = temporal_split(df)

    print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}\n")
    splits = {
        "Train": (X_train, y_train, None),
        "Val": (X_val, y_val, None),
        "Test": (X_test, y_test, None),
    }

    if not skip_baseline:
        train_baseline(X_train, y_train, splits)
        print()

    model = train_xgb(X_train, y_train, X_val, y_val, splits)

    # Calibrated intervals from a second, quantile model (see intervals.py)
    interval_model = train_quantile_model(X_train, y_train, X_val, y_val, model_params())

    # Test-set predictions, joined back to player identity for the app
    results = test_rows.copy()
    results["pred_log"] = model.predict(X_test)
    results["pred_value"] = np.exp(results["pred_log"])
    results["error_pct"] = (
        (results["pred_value"] - results["market_value"]) / results["market_value"] * 100
    )
    results = results.join(predict_intervals(interval_model, test_rows))
    cov = coverage(results, results["market_value"])
    print(f"\n=== INTERVALS (test) ===\n  coverage: 50% interval {cov['50%']:.0%}, 80% interval {cov['80%']:.0%}"
          f"   (conformal margins, log: {interval_model['margins']})")

    MODELS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLS}, f)
    with open(MODELS_DIR / "xgb_intervals.pkl", "wb") as f:
        pickle.dump({**interval_model, "features": FEATURE_COLS}, f)
    results.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)

    print(f"\nSaved models/xgb_model.pkl, models/xgb_intervals.pkl and "
          f"data/test_predictions.parquet ({len(results)} rows)")
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-baseline", action="store_true", help="skip the Ridge comparison")
    main(**vars(parser.parse_args()))
