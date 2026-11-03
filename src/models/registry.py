"""Model tracking and registry (MLflow, local).

Every training run is logged to MLflow — hyperparameters, metrics overall and
per league, interval coverage, the git commit, a fingerprint of the data — and
its model is registered as a new version of `transfer-edge-value`. One version
carries the `champion` alias; `src/models/promote.py` decides which.

The app never talks to MLflow: `export_champion()` copies the champion's files
to models/ and data/, where the app and the deploy bundle already read them.

Everything lives in mlops/ (gitignored): mlflow.db and artifacts/.

    mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db     # browse runs
"""

import hashlib
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402  (reads the settings above at import)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from src.features.engineer import TARGET, model_features, select_features  # noqa: E402
from src.models.intervals import coverage, predict_intervals  # noqa: E402

ROOT = Path(__file__).parents[2]
MLOPS_DIR = ROOT / "mlops"
TRACKING_URI = f"sqlite:///{MLOPS_DIR / 'mlflow.db'}"
EXPERIMENT = "transfer-edge"
MODEL_NAME = "transfer-edge-value"
CHAMPION = "champion"
BUNDLE = "bundle"   # artifact folder holding the files the app reads

# Champion files -> where the app expects them.
EXPORTS = {
    "xgb_model.pkl": ROOT / "models" / "xgb_model.pkl",
    "xgb_intervals.pkl": ROOT / "models" / "xgb_intervals.pkl",
    "test_predictions.parquet": ROOT / "data" / "test_predictions.parquet",
}


def setup() -> MlflowClient:
    """Point MLflow at mlops/ and select the experiment (created on first use)."""
    (MLOPS_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(MLOPS_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


# ── Metrics ──────────────────────────────────────────────────────────────────

def _scores(y_log, pred_log) -> dict:
    """R², RMSE(log) and median absolute % error in EUR — the metrics train.py prints."""
    y_log, pred_log = np.asarray(y_log), np.asarray(pred_log)
    ape = np.abs(np.exp(pred_log) - np.exp(y_log)) / np.exp(y_log)
    return {
        "r2": float(r2_score(y_log, pred_log)) if len(y_log) > 1 else float("nan"),
        "rmse_log": float(np.sqrt(np.mean((y_log - pred_log) ** 2))),
        "mape": float(np.median(ape) * 100),
        "n": float(len(y_log)),
    }


def evaluate(model, interval_model, rows: pd.DataFrame, prefix: str = "test") -> dict:
    """Metrics on `rows` (a temporal_split slice), overall and per league.

    Each model is fed its own features, so a champion trained on an older feature
    set is scored on exactly the same players as a new candidate.
    Keys look like `test_mape`, `test_mape.GB1`, `test_cov50`, `test_cov80.ES1`.
    """
    rows = rows[rows[TARGET].notna()]
    pred_log = model.predict(select_features(rows, model_features(model)))
    leagues = rows["league_code"] if "league_code" in rows.columns else pd.Series("ES1", index=rows.index)
    intervals = predict_intervals(interval_model, rows) if interval_model is not None else None

    out = {}
    groups = [("", np.ones(len(rows), dtype=bool))] + [
        (f".{lg}", (leagues == lg).to_numpy()) for lg in sorted(leagues.unique())]
    for suffix, mask in groups:
        for name, value in _scores(rows[TARGET].to_numpy()[mask], pred_log[mask]).items():
            out[f"{prefix}_{name}{suffix}"] = value
        if intervals is not None:
            cov = coverage(intervals[mask], rows["market_value"][mask])
            out[f"{prefix}_cov50{suffix}"] = cov["50%"] * 100
            out[f"{prefix}_cov80{suffix}"] = cov["80%"] * 100
    return out


# ── Provenance ───────────────────────────────────────────────────────────────

def git_commit() -> str:
    """Current commit, with '-dirty' when the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the modelling data: same panel -> same fingerprint."""
    h = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:12]


# ── Logging and registering ─────────────────────────────────────────────────

def log_and_register(model, interval_model, params: dict, metrics: dict,
                     test_predictions: pd.DataFrame, fingerprint: str,
                     description: str = "", tags: dict | None = None) -> str:
    """Log one training run and register its model. Returns the new version number."""
    setup()
    features = model_features(model)
    with mlflow.start_run(run_name=description or None) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})
        mlflow.set_tags({"git_commit": git_commit(), "data_fingerprint": fingerprint,
                         "n_features": str(len(features)), **(tags or {})})
        mlflow.log_dict({"features": features}, "features.json")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with open(tmp / "xgb_model.pkl", "wb") as f:
                pickle.dump({"model": model, "features": features}, f)
            with open(tmp / "xgb_intervals.pkl", "wb") as f:
                pickle.dump({**interval_model, "features": features}, f)
            test_predictions.to_parquet(tmp / "test_predictions.parquet", index=False)
            mlflow.log_artifacts(str(tmp), artifact_path=BUNDLE)

        info = mlflow.xgboost.log_model(model, name="model", registered_model_name=MODEL_NAME)
    version = str(info.registered_model_version)
    client = MlflowClient()
    if description:
        client.update_model_version(MODEL_NAME, version, description=description)
    print(f"logged run {run.info.run_id[:8]} -> registered {MODEL_NAME} version {version}")
    return version


# ── Reading versions back ───────────────────────────────────────────────────

def champion_version() -> str | None:
    setup()
    try:
        return MlflowClient().get_model_version_by_alias(MODEL_NAME, CHAMPION).version
    except mlflow.exceptions.MlflowException:
        return None


def latest_version() -> str | None:
    setup()
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
    return max((v.version for v in versions), key=int, default=None)


def load_version(version: str) -> tuple:
    """(model, interval_model) of a registered version, from its run's bundle."""
    client = setup()
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    folder = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=BUNDLE))
    with open(folder / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)["model"]
    with open(folder / "xgb_intervals.pkl", "rb") as f:
        intervals = pickle.load(f)
    return model, intervals


def set_champion(version: str) -> None:
    setup()
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION, version)


def export_champion() -> str:
    """Copy the champion's model files to models/ and data/. Returns its version."""
    version = champion_version()
    if version is None:
        raise RuntimeError(f"no '{CHAMPION}' alias on {MODEL_NAME} yet — run src.models.promote")
    client = setup()
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    folder = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=BUNDLE))
    for name, dest in EXPORTS.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / name, dest)
    print(f"exported champion (version {version}) -> models/, data/test_predictions.parquet")
    return version


# ── Registering a model trained before tracking existed ─────────────────────

def register_files(description: str) -> str:
    """Register the model currently in models/ (e.g. v1, trained before MLflow).

    Metrics are computed now, on the current data, exactly as for a new run.
    """
    from src.data.loader import load_panel
    from src.features.engineer import FEATURE_COLS, build_features, temporal_split
    from src.models.train import model_params

    with open(EXPORTS["xgb_model.pkl"], "rb") as f:
        model = pickle.load(f)["model"]
    with open(EXPORTS["xgb_intervals.pkl"], "rb") as f:
        intervals = pickle.load(f)
    predictions = pd.read_parquet(EXPORTS["test_predictions.parquet"])

    df = build_features(load_panel())
    _, (_, _, val_rows), (_, _, test_rows) = temporal_split(df)
    metrics = {**evaluate(model, intervals, val_rows, prefix="val"),
               **evaluate(model, intervals, test_rows, prefix="test")}
    fingerprint = data_fingerprint(df[["year", "player_id", TARGET, *FEATURE_COLS]])
    return log_and_register(model, intervals, params=model_params(), metrics=metrics,
                            test_predictions=predictions, fingerprint=fingerprint,
                            description=description, tags={"source": "models/ files"})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register the model files in models/ as a new version.")
    parser.add_argument("--register-files", metavar="DESCRIPTION", required=True,
                        help="e.g. 'v1.1: LaLiga, 37 features'")
    register_files(parser.parse_args().register_files)
