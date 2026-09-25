import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"


def load_panel() -> pd.DataFrame:
    """The merged player panel (TM + Understat), plus FBref stats when matched.

    `fbref_matched.parquet` comes from notebook 05. Without it the fb_* columns
    are simply absent and the FBref features come out NaN.
    """
    panel = pd.read_parquet(DATA_DIR / "panel_final.parquet")
    fbref = DATA_DIR / "fbref_matched.parquet"
    if fbref.exists():
        matched = pd.read_parquet(fbref).drop(columns=["fbref_match_score"])
        panel = panel.assign(year=panel["year"].astype(int), player_id=panel["player_id"].astype(str))
        matched = matched.assign(year=matched["year"].astype(int), player_id=matched["player_id"].astype(str))
        panel = panel.merge(matched, on=["year", "player_id"], how="left")
    return panel


def load_players() -> pd.DataFrame:
    """Load raw Transfermarkt player panel before merge."""
    return pd.read_parquet(DATA_DIR / "players_panel.parquet")


def load_understat() -> pd.DataFrame:
    """Load raw Understat stats before merge."""
    return pd.read_parquet(DATA_DIR / "understat_stats.parquet")


def load_predictions() -> pd.DataFrame:
    """Load test-set predictions (2023–2024)."""
    return pd.read_parquet(DATA_DIR / "test_predictions.parquet")
