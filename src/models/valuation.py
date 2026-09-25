"""Score the whole panel with the trained model — the app's data layer.

Everything the Streamlit app shows comes from `score_panel()`: one row per
player-season, with the model's valuation and undervalue score attached.
Keeping this here (not in app/main.py) means the numbers on screen are
produced by the same code path the tests and notebooks use.
"""

import numpy as np
import pandas as pd

from src.data.loader import load_panel, load_predictions
from src.features.engineer import (
    EXCLUDED_POSITIONS,
    TRAIN_END,
    build_features,
    compute_undervalue_score,
)
from src.models.intervals import predict_intervals
from src.models.predict import predict

# Fallback if data/test_predictions.parquet is missing.
DEFAULT_MAPE = 0.25


def score_panel(model, interval_model=None) -> pd.DataFrame:
    """Every valued player-season, with `pred_value` and `undervalue_score`.

    Features are built on the full panel first (percentiles and club tier are
    relative to it), then goalkeepers are dropped — the model was never trained
    on them, so any number it gave would be meaningless.
    """
    df = build_features(load_panel())
    df = df[~df["pos_group"].isin(EXCLUDED_POSITIONS)].reset_index(drop=True)
    df["pred_value"] = predict(model, df)
    df["undervalue_score"] = compute_undervalue_score(df)
    if interval_model is not None:
        df = df.join(predict_intervals(interval_model, df))
    df["in_training"] = df["year"] <= TRAIN_END
    if "market_value_imputed" not in df.columns:
        df["market_value_imputed"] = False
    return df


def top_undervalued(df: pd.DataFrame, position_group=None, limit=20,
                    min_market_value=0) -> pd.DataFrame:
    """Highest undervalue scores, optionally within one position group.

    Rows whose market value was interpolated from neighbouring seasons are left
    out: the score would be measured against a number Transfermarkt never listed.

    `min_market_value` exists because the score is a ratio: academy players sit
    at Transfermarkt's EUR 25k floor, so a EUR 3M prediction reads as "13,600%
    undervalued" and buries every real candidate. A floor of ~EUR 1M keeps the
    list to players a club would actually scout.
    """
    rows = df[~df["market_value_imputed"].fillna(False).astype(bool)]
    rows = rows[rows["market_value"] >= min_market_value]
    if position_group is not None:
        rows = rows[rows["pos_group"] == position_group]
    return rows.nlargest(limit, "undervalue_score")


def test_mape() -> float:
    """Median absolute % error on the held-out test seasons, as a fraction."""
    try:
        return float(load_predictions()["error_pct"].abs().median() / 100)
    except FileNotFoundError:
        return DEFAULT_MAPE


def player_history(df: pd.DataFrame, player_id) -> pd.DataFrame:
    """All scored seasons for one player, oldest first."""
    return df[df["player_id"] == player_id].sort_values("year")


def interval_coverage() -> dict | None:
    """Share of test-season players whose listed value fell inside each interval."""
    try:
        t = load_predictions()
    except FileNotFoundError:
        return None
    if "low_80" not in t.columns:
        return None
    return {tag: float(((t["market_value"] >= t[f"low_{tag}"]) & (t["market_value"] <= t[f"high_{tag}"])).mean())
            for tag in (50, 80)}


# (label, season total column, per-90 column or None). Understat totals first,
# then FBref's. Per 90 uses each source's own minutes.
SEASON_STATS = [
    ("Appearances", "games", None),
    ("Minutes", "minutes", None),
    ("Goals", "goals", "goals_p90"),
    ("Expected goals (xG)", "xG", "xG_p90"),
    ("Non-penalty xG", "npxG", "npxG_p90"),
    ("Shots", "shots", "shots_p90"),
    ("Assists", "assists", "assists_p90"),
    ("Expected assists (xA)", "xA", "xA_p90"),
    ("Key passes", "key_passes", "key_passes_p90"),
    ("xG chain", "xGChain", "xGChain_p90"),
    ("xG buildup", "xGBuildup", "xGBuildup_p90"),
    ("Tackles won", "fb_tackles_won", "tackles_won_p90"),
    ("Interceptions", "fb_interceptions", "interceptions_p90"),
    ("Fouls drawn", "fb_fouled", "fouled_p90"),
    ("Fouls committed", "fb_fouls", "fouls_p90"),
    ("Crosses", "fb_crosses", "crosses_p90"),
    ("Yellow cards", "yellow_cards", None),
]

# Peers need enough minutes for their per-90 numbers to mean something.
PEER_MIN_MINUTES = 450


def season_stats(df: pd.DataFrame, player: pd.Series) -> tuple[pd.DataFrame, int]:
    """One player's season stats with a percentile against his positional peers.

    Peers are players of the same position group in the same season with at
    least PEER_MIN_MINUTES minutes. The percentile ranks the per-90 rate where
    there is one (so a starter and a sub compare fairly), else the season total.
    Returns the table and the number of peers.
    """
    peers = df[(df["year"] == player["year"]) & (df["pos_group"] == player["pos_group"])
               & (df["minutes"] >= PEER_MIN_MINUTES)]
    rows = []
    for label, total_col, per90_col in SEASON_STATS:
        if total_col not in df.columns:
            continue
        rank_col = per90_col or total_col
        value = player.get(rank_col)
        pool = peers[rank_col].dropna() if rank_col in peers else pd.Series(dtype=float)
        pct = (np.nan if pd.isna(value) or pool.empty
               else 100 * ((pool < value).mean() + 0.5 * (pool == value).mean()))
        rows.append({"stat": label, "total": player.get(total_col),
                     "per_90": player.get(per90_col) if per90_col else np.nan,
                     "percentile": pct})
    return pd.DataFrame(rows), len(peers)
