"""Feature engineering for the market-value model.

`build_features()` is the single source of truth for how raw panel rows become
model inputs. It was extracted verbatim from notebook 03 so that training and
inference share one code path — if you change a transformation here, retrain.

IMPORTANT: several features (percentile ranks, club tier) are computed *relative
to the rest of the panel*. Always call `build_features()` on the full panel and
then select the rows you care about. Calling it on a single player's row will
silently produce a 100th-percentile, tier-0 player.

MODELLING SCOPE — two decisions made from EDA, both enforced downstream of
`build_features()` rather than inside it, so the function stays a general-purpose
transform usable for EDA on the full panel:

- **Goalkeepers are excluded from training** (`temporal_split()`). Understat's
  outfield stats (xG, xA, key passes, ...) carry no signal for a GK's value, and
  GK is the second-smallest position group (706 rows) — not enough to support a
  dedicated model yet. `posgrp_GK` is not in `FEATURE_COLS`. The app cannot score
  goalkeepers until a GK-specific model exists.
- **Defensive Midfield is merged into DEF**, not kept separate and not merged
  into MID. `simplify_position()` maps it there directly. DM alone was only 399
  rows — too thin for reliable within-position percentiles — and centre-back /
  DM roles overlap more (positioning, defensive actions) than DM / attacking MID
  do. `posgrp_DM` no longer exists.

NO LEAKAGE — every feature must be knowable when the target value is set:

- The target is the market value at the END of the season (notebook 01), so the
  full-season Understat stats genuinely precede it.
- `club_med_value` is the median of the player's *teammates*, never including
  his own value (that would put the target inside a feature).
- Interpolated market values (`market_value_imputed`) are midpoints built from
  the *next* season's value. They are never a training/test target
  (`temporal_split()`), never a lag, and never enter a club median.
- Anything fitted from the target — e.g. `fit_peak_ages()`, used by the EDA —
  sees training seasons only, never validation or test targets.
"""

import numpy as np
import pandas as pd

# Walk-forward split. Anything fitted from the target (peak ages) may only look
# at seasons <= TRAIN_END.
TRAIN_END = 2022
VAL_YEAR = 2023

# Prior peak ages by position — the conventional football wisdom. fit_peak_ages()
# re-estimates them on the training seasons (the EDA compares the two); these are
# its fallback when a group has too little data. No model feature uses them: age
# is the only age feature, because trees draw the rise-and-fall curve themselves.
PEAK_AGES = {"GK": 30, "DEF": 28, "MID": 27, "ATT": 26, "Unknown": 27}

# Fallback peak age for any position group not listed above
DEFAULT_PEAK_AGE = 27

FOOT_MAP = {"right": 1, "left": 2, "both": 3}

# Raw Understat counting stats that get converted to per-90 rates
PER90_STATS = [
    "goals", "xG", "assists", "xA", "shots",
    "key_passes", "npxG", "xGChain", "xGBuildup",
]

# FBref season counts (notebook 05 -> fbref_matched.parquet) turned into per-90
# rates over FBref's own minutes. The only defensive stats FBref still publishes.
FBREF_PER90 = ["tackles_won", "interceptions", "fouls", "fouled", "crosses"]

# Columns ranked to a within-(year, position) percentile
PERCENTILE_STATS = ["xG_p90", "xA_p90", "npxG_p90", "minutes", "def_actions_p90"]

# Understat codes a season's positions as a multi-label string like "D F M S".
# "S" marks a substitute appearance, not a position, so it never counts.
POSITION_TOKENS = frozenset({"D", "M", "F", "GK"})

# Columns lagged one season (only when the previous season is consecutive)
LAG_STATS = ["log_value", "xG_p90", "goals_p90", "assists_p90", "minutes"]

TARGET = "log_value"

# Positions excluded from training — see the module docstring. Applied in
# temporal_split(), not build_features(), so the full panel (GK included)
# stays available for EDA and any future GK-specific model.
EXCLUDED_POSITIONS = {"GK"}

# The 47 features the model is trained on — order must match training.
# (46 -> 44: posgrp_DM removed by the DM/DEF merge, posgrp_GK removed because
# goalkeepers are excluded from training — see the module docstring.)
FEATURE_COLS = [
    # Personal
    "age", "height", "foot_enc",
    # Per-90 stats
    "minutes", "games",
    "goals_p90", "xG_p90", "assists_p90", "xA_p90", "shots_p90",
    "key_passes_p90", "npxG_p90", "xGChain_p90", "xGBuildup_p90",
    # Engineered
    "goals_minus_xG", "assists_minus_xA",
    "xG_p90_pct", "xA_p90_pct", "npxG_p90_pct", "minutes_pct",
    "position_versatility",
    "club_med_value", "club_tier",
    # FBref (NaN when the player-season has no FBref match)
    "tackles_won_p90", "interceptions_p90", "fouls_p90", "fouled_p90", "crosses_p90",
    "def_actions_p90_pct",
    # Lag features (NaN when year_diff != 1)
    "log_value_lag1", "xG_p90_lag1", "goals_p90_lag1",
    "assists_p90_lag1", "minutes_lag1", "value_growth",
    # Position one-hot (ATT dropped as reference — alphabetically first;
    # GK excluded from training, DM merged into DEF — see module docstring)
    "posgrp_DEF", "posgrp_MID",
    # Nationality one-hot (Argentina dropped as reference — it is the
    # alphabetically first category, which is what drop_first=True removes)
    "natgrp_Brazil", "natgrp_Colombia", "natgrp_France", "natgrp_Morocco",
    "natgrp_Other", "natgrp_Portugal", "natgrp_Senegal", "natgrp_Serbia",
    "natgrp_Spain", "natgrp_Uruguay",
]


def simplify_position(pos) -> str:
    """Map a verbose Transfermarkt position string to one of 4 groups.

    Defensive Midfield folds into DEF (not MID) — see the module docstring for
    why. Check it before the generic "midfield" branch, since "defensive
    midfield" would otherwise match that first.
    """
    if not isinstance(pos, str):
        return "Unknown"
    p = pos.lower()
    if "goalkeeper" in p:
        return "GK"
    if "defender" in p or "defensive midfield" in p:
        return "DEF"
    if "midfield" in p:
        return "MID"
    if "attack" in p or "winger" in p:
        return "ATT"
    return "Unknown"


def position_versatility(code) -> float:
    """How many distinct positions Understat recorded for a player-season.

    `position_us` is a multi-label code such as "D F M S" — this player turned
    out in defence, attack and midfield, and also came off the bench. "S" is a
    substitute appearance rather than a position, so it is excluded.

    Returns NaN when the row never matched Understat, and 0.0 for a player whose
    only code is "S" (appeared, but never in a recorded position). Goalkeepers
    are always exactly 1, so the feature carries no signal for them.
    """
    if not isinstance(code, str):
        return np.nan
    return float(len(POSITION_TOKENS.intersection(code.split())))


def _imputed(df: pd.DataFrame) -> pd.Series:
    """True where market_value was interpolated in notebook 01."""
    if "market_value_imputed" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["market_value_imputed"].fillna(False).astype(bool)


def fit_peak_ages(df: pd.DataFrame, train_end: int = TRAIN_END, min_rows: int = 10) -> dict:
    """Age at which median log(market_value) peaks, per position — training seasons only.

    Fitted from the target, so it must never see validation or test seasons.
    The per-age medians are smoothed over a 3-year window first: near the top
    the curve is flat, and the raw argmax jumps by several years on noise.
    Groups with too little data keep their `PEAK_AGES` prior.
    """
    train = df[(df["year"] <= train_end) & ~_imputed(df) & df["age"].notna()]
    log_value = np.log(train["market_value"])
    peaks = dict(PEAK_AGES)
    for group, idx in train.groupby("pos_group").groups.items():
        by_age = log_value.loc[idx].groupby(train.loc[idx, "age"].round()).agg(["median", "size"])
        by_age = by_age[by_age["size"] >= min_rows]
        if len(by_age) < 3:
            continue
        smooth = by_age["median"].rolling(3, center=True, min_periods=1).mean()
        peaks[group] = int(smooth.idxmax())
    return peaks


def _teammates_median(df: pd.DataFrame) -> pd.Series:
    """Median market value of each player's teammates that season, excluding him.

    Interpolated values are left out of the pool too — they carry next season's
    value. A player with no valued teammate gets NaN.
    """
    values = df["market_value"].to_numpy(dtype=float)
    in_pool = ~_imputed(df).to_numpy()
    out = np.full(len(df), np.nan)
    for rows in df.groupby(["year", "club"]).indices.values():
        pool = rows[in_pool[rows]]
        for i in rows:
            others = values[pool[pool != i]]
            if len(others):
                out[i] = np.median(others)
    return pd.Series(out, index=df.index)


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw merged panel into a modelling frame.

    Returns a copy with `log_value` (the target), every column in
    `FEATURE_COLS`, and the helper columns `pos_group` / `nat_group` kept for
    grouping and display. Rows with a missing or non-positive market value are
    dropped, since the target is log(market_value).
    """
    df = df_raw.copy()
    df["year"] = df["year"].astype(int)

    # ── Target ───────────────────────────────────────────────────────────────
    df = df[df["market_value"].notna() & (df["market_value"] > 0)].reset_index(drop=True)
    df["log_value"] = np.log(df["market_value"])

    # ── Per-90 stats ─────────────────────────────────────────────────────────
    nineties = df["minutes"] / 90
    for col in PER90_STATS:
        df[f"{col}_p90"] = df[col] / nineties

    # FBref: per 90 over FBref's own minutes, so numerator and denominator
    # come from the same source.
    fb_nineties = df["fb_minutes_90s"].where(df["fb_minutes_90s"] > 0) if "fb_minutes_90s" in df else np.nan
    for col in FBREF_PER90:
        df[f"{col}_p90"] = df[f"fb_{col}"] / fb_nineties if f"fb_{col}" in df else np.nan
    df["def_actions_p90"] = df["tackles_won_p90"] + df["interceptions_p90"]

    # ── Personal attributes ──────────────────────────────────────────────────
    df["foot_enc"] = df["foot"].map(FOOT_MAP).fillna(0).astype(int)

    df["pos_group"] = df["position"].apply(simplify_position)
    pos_dummies = pd.get_dummies(df["pos_group"], prefix="posgrp", drop_first=True).astype(int)
    df = pd.concat([df, pos_dummies], axis=1)

    # Positional versatility — a player who covers several roles is worth more
    # than the same output from a specialist, and it holds within minutes band.
    df["position_versatility"] = (
        df["position_us"].apply(position_versatility)
        if "position_us" in df.columns else np.nan
    )

    top_nat = df["nationality"].value_counts().head(10).index
    df["nat_group"] = df["nationality"].apply(lambda x: x if x in top_nat else "Other")
    nat_dummies = pd.get_dummies(df["nat_group"], prefix="natgrp", drop_first=True).astype(int)
    df = pd.concat([df, nat_dummies], axis=1)

    # ── 1. xG overperformance (clinical finishers) ───────────────────────────
    df["goals_minus_xG"] = df["goals"] - df["xG"]
    df["assists_minus_xA"] = df["assists"] - df["xA"]

    # ── 2. Percentile within year + position ─────────────────────────────────
    # 0.4 xG/90 is median for a striker and impossible for a keeper.
    for col in PERCENTILE_STATS:
        df[f"{col}_pct"] = (
            df.groupby(["year", "pos_group"])[col].rank(pct=True, na_option="keep")
        )

    # (No age transforms: age2, years_to_peak and prime_window used to live here.
    # For trees they are the same information as `age` — any split on age² or on
    # peak-age-minus-age is a split on age — and they only divided the age effect
    # across three SHAP bars. Removing them cost no measurable accuracy.)

    # ── 4. Club tier ─────────────────────────────────────────────────────────
    # Real Madrid / Barça players get visibility that inflates value.
    df["club_med_value"] = _teammates_median(df)
    df["club_tier"] = df.groupby("year")["club_med_value"].transform(
        lambda x: pd.qcut(x, q=5, labels=False, duplicates="drop")
    ).astype(float)

    # ── 5. Lag features (previous season's value and stats) ──────────────────
    # Single strongest predictor: what the market paid last year.
    df = df.sort_values(["player_id", "year"]).reset_index(drop=True)
    year_diff = df.groupby("player_id")["year"].diff()
    prev_imputed = _imputed(df).groupby(df["player_id"]).shift(1, fill_value=False)
    for col in LAG_STATS:
        lagged = df.groupby("player_id")[col].shift(1)
        # Only use the lag if it is the immediately preceding season (no gaps)
        df[f"{col}_lag1"] = lagged.where(year_diff == 1)
    # An interpolated value is half next season's value: as a lag it would hand
    # this season's target to the model. Stats lags are real data and stay.
    df.loc[prev_imputed, "log_value_lag1"] = np.nan

    df["value_growth"] = df["log_value"] - df["log_value_lag1"]

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the model matrix, in `FEATURE_COLS` order.

    Reindexing guards against one-hot drift: `pd.get_dummies` only emits columns
    for categories present in the data, so a subset missing (say) Senegalese
    players would otherwise produce a narrower matrix than the model expects.
    Missing one-hot columns become 0; missing numeric columns become NaN, which
    XGBoost handles natively.
    """
    X = df.reindex(columns=FEATURE_COLS)
    onehot = [c for c in FEATURE_COLS if c.startswith(("posgrp_", "natgrp_"))]
    X[onehot] = X[onehot].fillna(0).astype(int)
    return X


def temporal_split(df: pd.DataFrame, train_end=TRAIN_END, val_year=VAL_YEAR):
    """Walk-forward split: train ≤TRAIN_END (2022), validate on VAL_YEAR (2023), test after (2024–2025).

    Drops `EXCLUDED_POSITIONS` (goalkeepers) and rows whose market value was
    interpolated — a target built from the next season's value is leakage in
    training and a free point in evaluation. This is the one place both gates
    are enforced, so `df` passed in should be the full `build_features()` output.

    Returns three (X, y, rows) tuples. `rows` is the matching slice of `df`,
    so predictions can be joined back to player names and clubs.
    """
    df = df[~df["pos_group"].isin(EXCLUDED_POSITIONS) & ~_imputed(df)]

    def _xy(subset):
        mask = subset[TARGET].notna()
        subset = subset[mask]
        return select_features(subset), subset[TARGET], subset

    return (
        _xy(df[df["year"] <= train_end]),
        _xy(df[df["year"] == val_year]),
        _xy(df[df["year"] > val_year]),
    )


def compute_undervalue_score(df: pd.DataFrame) -> pd.Series:
    """(predicted - market) / market. Positive = undervalued.

    A fraction, not a percentage — `test_predictions.parquet` stores the same
    quantity ×100 in its `error_pct` column.
    """
    return (df["pred_value"] - df["market_value"]) / df["market_value"]
