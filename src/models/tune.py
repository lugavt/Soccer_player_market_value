"""Tune XGBoost hyperparameters with Optuna, without touching the test seasons.

    python -m src.models.tune            # 100 trials, writes models/best_params.json
    python -m src.models.tune --trials 30

Validation is walk-forward over the seasons before the test period: train on
2017..Y-1, validate on Y, for Y = 2019..VAL_YEAR, and minimise the mean RMSE
(log) across those folds. Tuning on the single validation season would fit its
noise; tuning on the test seasons would make the test number meaningless.

The current hand-set XGB_PARAMS are scored on the same folds, so the output
says whether tuning actually beat them. train.py picks up best_params.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb

from src.data.loader import load_panel
from src.features.engineer import (
    EXCLUDED_POSITIONS, TARGET, VAL_YEAR, _imputed, build_features, select_features,
)
from src.models.train import XGB_PARAMS

MODELS_DIR = Path(__file__).parents[2] / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
FIRST_VAL_YEAR = 2019          # a fold needs at least two training seasons
MAX_TREES = 3000
EARLY_STOPPING = 50


def modelling_rows():
    """Same gates as temporal_split(): no goalkeepers, no interpolated targets, no test seasons."""
    df = build_features(load_panel())
    df = df[~df["pos_group"].isin(EXCLUDED_POSITIONS) & ~_imputed(df) & df[TARGET].notna()]
    return df[df["year"] <= VAL_YEAR]


def make_folds(df):
    folds = []
    for val_year in range(FIRST_VAL_YEAR, VAL_YEAR + 1):
        tr, va = df[df["year"] < val_year], df[df["year"] == val_year]
        folds.append((select_features(tr), tr[TARGET], select_features(va), va[TARGET]))
    return folds


def cv_rmse(params: dict, folds) -> tuple[float, int]:
    """Mean validation RMSE(log) over the folds, and the mean best iteration."""
    scores, iters = [], []
    for X_tr, y_tr, X_va, y_va in folds:
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_va)
        scores.append(float(np.sqrt(np.mean((y_va - pred) ** 2))))
        iters.append(model.best_iteration)
    return float(np.mean(scores)), int(np.mean(iters))


def objective(trial, folds):
    params = dict(
        n_estimators=MAX_TREES,
        early_stopping_rounds=EARLY_STOPPING,
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
        max_depth=trial.suggest_int("max_depth", 3, 8),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1, 30, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 2.0),
    )
    score, iters = cv_rmse(params, folds)
    trial.set_user_attr("mean_best_iteration", iters)
    return score


def main(trials: int = 100):
    folds = make_folds(modelling_rows())
    print(f"{len(folds)} walk-forward folds, validating on {FIRST_VAL_YEAR}–{VAL_YEAR}; "
          f"test seasons untouched")

    baseline, _ = cv_rmse(XGB_PARAMS, folds)
    print(f"current XGB_PARAMS: CV RMSE(log) = {baseline:.4f}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective(t, folds), n_trials=trials,
                   callbacks=[lambda s, t: print(f"  trial {t.number + 1:3}/{trials}  "
                                                 f"{t.value:.4f}  best {s.best_value:.4f}", flush=True)])

    best = study.best_trial
    print(f"\nbest CV RMSE(log) = {best.value:.4f}  (current {baseline:.4f}, "
          f"{100 * (best.value - baseline) / baseline:+.1f}%)")
    print("best params:", best.params)

    MODELS_DIR.mkdir(exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps({
        "params": best.params,
        "cv_rmse_log": best.value,
        "cv_rmse_log_current_params": baseline,
        "trials": trials,
        "folds": f"walk-forward, validation seasons {FIRST_VAL_YEAR}-{VAL_YEAR}",
    }, indent=2))
    print(f"saved {BEST_PARAMS_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    main(**vars(parser.parse_args()))
