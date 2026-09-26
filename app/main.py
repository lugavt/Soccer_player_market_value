"""Transfer Edge -- La Liga player valuation with explainable predictions.

Two tabs: look up a player in the current season and see why the model prices
him where he is and how his value has moved over the seasons on record, or
browse the most undervalued players by position.

All the logic lives in src/ (build_features, the model, scoring); this file
is presentation only.

    streamlit run app/main.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# `streamlit run app/main.py` puts app/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineer import select_features  # noqa: E402
from src.models.predict import explain, load_interval_model, load_model  # noqa: E402
from src.models.valuation import (  # noqa: E402
    interval_coverage,
    TRAIN_END,
    player_history,
    season_stats,
    score_panel,
    test_mape,
    top_undervalued,
)

# Diverging pair (blue <-> red) plus one categorical slot for the market-value
# reference. Blue/red is colourblind-safe; green/red is not.
COLOR_UP = "#2a78d6"
COLOR_DOWN = "#e34948"
COLOR_MARKET = "#eb6834"
COLOR_MUTED = "#52514e"

# Goalkeepers are not here on purpose: the model is trained on outfield stats
# only. Defensive midfielders are folded into DEF (see src/features/engineer.py).
POSITION_LABELS = {
    "DEF": "Defender (incl. defensive midfield)",
    "MID": "Midfield",
    "ATT": "Attack",
}

MIN_VALUE_OPTIONS = [0, 250_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

# Plain-language definitions for engineered feature names that show up in the
# "Why this valuation" breakdown but aren't self-explanatory on their own.
FEATURE_GLOSSARY = {
    "log_value_lag1": (
        "The player's market value from the previous season (log-space), "
        "when available -- usually the single strongest predictor."
    ),
    "value_growth": (
        "Year-over-year change in market value (log-space), vs. the player's "
        "value the previous season."
    ),
    "age": (
        "Age at the end of the season. The model learns the rise-and-fall curve "
        "itself: value tends to climb into the mid-20s and fall after."
    ),
    "club_med_value": (
        "Median market value of all players at the player's club that season -- "
        "a proxy for the club's prestige/visibility."
    ),
    "club_tier": "Which fifth of La Liga the club falls into by squad value that season.",
    "minutes_pct": (
        "Percentile rank of minutes played, compared to other players in the "
        "same position and season (0 = fewest, 1 = most)."
    ),
    "position_versatility": (
        "How many distinct positions (defence, midfield, attack) the player "
        "was used in that season. Versatile players carry a premium."
    ),
    "natgrp_Spain": (
        "Whether the player's nationality group is Spain (1) or not (0), one "
        "of the model's nationality indicator features."
    ),
}

FAVICON = Path(__file__).parent / "assets" / "favicon.png"

st.set_page_config(page_title="Transfer Edge", page_icon=str(FAVICON), layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_interval_model():
    return load_interval_model()


@st.cache_data
def get_players(_model, _interval_model) -> pd.DataFrame:
    return score_panel(_model, _interval_model)


@st.cache_data
def get_coverage() -> dict | None:
    return interval_coverage()


@st.cache_data
def get_mape() -> float:
    return test_mape()


def format_eur(value: float | None) -> str:
    """Compact euro formatting: 45_000_000 -> 'EUR 45.0M'."""
    if value is None or pd.isna(value):
        return "unknown"
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}K"
    return f"EUR {value:.0f}"


def format_height(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "height unknown"
    return f"{value:.2f} m" if value < 3 else f"{value:.0f} cm"


def format_feature_value(value) -> str:
    if value is None or pd.isna(value):
        return "missing"
    return f"{value:,.3g}"


def valuation_range_chart(predicted: float, market: float, low: float, high: float) -> alt.LayerChart:
    """Point estimate with its calibrated 50% range, against the listed market value."""
    band = pd.DataFrame([{"low": low, "high": high, "predicted": predicted, "market": market}])

    lo, hi = min(low, market), max(high, market)
    pad = max((hi - lo) * 0.2, hi * 0.05)
    # clamp at zero -- a negative euro axis reads as a bug
    scale = alt.Scale(domain=[max(0.0, lo - pad), hi + pad])
    axis = alt.Axis(format="~s", title="Value (EUR)", grid=False, tickCount=6)

    def x(field):
        return alt.X(f"{field}:Q", scale=scale, axis=axis)

    range50 = (
        alt.Chart(band).mark_rule(size=14, color=COLOR_UP, opacity=0.3)
        .encode(x=x("low"), x2="high",
                tooltip=[alt.Tooltip("low:Q", title="50% range from", format=",.0f"),
                         alt.Tooltip("high:Q", title="50% range to", format=",.0f")])
    )
    point = (
        alt.Chart(band).mark_point(size=260, filled=True, color=COLOR_UP)
        .encode(x=x("predicted"), tooltip=[alt.Tooltip("predicted:Q", title="Model", format=",.0f")])
    )
    market_rule = (
        alt.Chart(band).mark_rule(size=3, color=COLOR_MARKET, strokeDash=[6, 4])
        .encode(x=x("market"), tooltip=[alt.Tooltip("market:Q", title="Listed", format=",.0f")])
    )
    predicted_label = (
        alt.Chart(band).mark_text(dy=-28, color=COLOR_UP, fontWeight="bold", fontSize=13)
        .encode(x=x("predicted"), text=alt.value(f"Model: {format_eur(predicted)}"))
    )
    market_label = (
        alt.Chart(band).mark_text(dy=28, color=COLOR_MARKET, fontWeight="bold", fontSize=13)
        .encode(x=x("market"), text=alt.value(f"Listed: {format_eur(market)}"))
    )
    return (range50 + market_rule + point + predicted_label + market_label).properties(
        height=170, padding={"top": 30, "bottom": 10, "left": 10, "right": 10},
    )


def shap_bar_chart(shap_df: pd.DataFrame) -> alt.LayerChart:
    """Every feature's SHAP contribution for one player, largest push first.

    Bars are in log-space: they add up exactly to (prediction - baseline) there,
    so their lengths are comparable with each other but are not euro amounts.
    """
    df = shap_df.sort_values("impact", ascending=False).copy()
    df["direction"] = np.where(df["impact"] >= 0, "Pushes value up", "Drags value down")
    order = df["feature"].tolist()

    bars = (
        alt.Chart(df)
        .mark_bar(height=11)
        .encode(
            y=alt.Y("feature:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=220, ticks=False, domain=False)),
            x=alt.X("impact:Q", title="SHAP value (log-space)",
                    axis=alt.Axis(grid=True, gridColor="#e1e0d9")),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Pushes value up", "Drags value down"],
                                range=[COLOR_UP, COLOR_DOWN]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("value_label:N", title="Player's value"),
                alt.Tooltip("impact:Q", title="SHAP (log)", format="+.3f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3c2b7").encode(x="x:Q")
    return (bars + zero).properties(height=16 * len(df))


def value_history_chart(history: pd.DataFrame) -> alt.LayerChart:
    """Listed market value vs model valuation, one point per season on record.

    Hollow orange points are market values interpolated from neighbouring
    seasons (Transfermarkt had no December snapshot), not real listings.
    """
    listed = history.assign(
        series="Listed market value", value=history["market_value"],
        hollow=history["market_value_imputed"].fillna(False).astype(bool),
    )
    model = history.assign(series="Model valuation", value=history["pred_value"], hollow=False)
    long = pd.concat([listed, model], ignore_index=True)[
        ["year", "series", "value", "hollow", "club", "in_training"]
    ]
    long["note"] = np.select(
        [long["hollow"], (long["series"] == "Model valuation") & long["in_training"]],
        ["interpolated -- no December listing",
         f"season in training data (<= {TRAIN_END}): fitted, not predicted"],
        default="",
    )

    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=["Listed market value", "Model valuation"],
                        range=[COLOR_MARKET, COLOR_UP]),
        legend=alt.Legend(orient="top", title=None),
    )
    x = alt.X("year:O", title="Season", axis=alt.Axis(labelAngle=0))
    y = alt.Y("value:Q", title="Value (EUR)", axis=alt.Axis(format="~s"))
    tooltip = [
        alt.Tooltip("year:O", title="Season"),
        alt.Tooltip("club:N", title="Club"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("value:Q", title="EUR", format=",.0f"),
        alt.Tooltip("note:N", title="Note"),
    ]

    base = alt.Chart(long).encode(x=x, y=y, color=color, tooltip=tooltip)
    lines = base.mark_line(strokeWidth=2)
    solid = base.transform_filter("!datum.hollow").mark_point(size=80, filled=True, opacity=1)
    hollow = base.transform_filter("datum.hollow").mark_point(
        size=80, filled=False, strokeWidth=2, opacity=1
    )
    return (lines + solid + hollow).properties(height=280)


with st.spinner("Loading the model and scoring every player-season..."):
    model = get_model()
    interval_model = get_interval_model()
    players = get_players(model, interval_model)
    mape = get_mape()
    cover = get_coverage()

latest_season = int(players["year"].max())

st.title("La Liga players valuation")
st.caption(
    "Check La Liga players' market value and compare with what we believe should "
    "be their actual value. Explore the features that push or drag them. Jump to "
    "the leaderboard to see some market gems."
)

search_tab, leaderboard_tab = st.tabs(["Player search", "Leaderboard"])

with search_tab:
    # Profiles are always the current season; earlier seasons only appear in
    # the value-over-time chart below.
    club_col, player_col = st.columns(2)
    season_players = players[players["year"] == latest_season]
    club_series = season_players["club"].fillna("Unknown club")
    selected_club = club_col.selectbox("Club", sorted(club_series.unique()), index=0)

    club_players = season_players[club_series == selected_club].sort_values("player_name")
    names = dict(zip(club_players["player_id"], club_players["player_name"].fillna("Unknown")))
    selected_id = player_col.selectbox("Player", list(names), format_func=names.get, index=0)

    row = club_players[club_players["player_id"] == selected_id].iloc[[0]]
    player = row.iloc[0]

    predicted = float(player["pred_value"])
    market = float(player["market_value"])
    score = player["undervalue_score"]

    meta_parts = [
        str(player.get("club") or ""),
        str(player.get("position") or ""),
        f"{player['age']:.0f} years old" if pd.notna(player["age"]) else "",
        format_height(player.get("height")),
        str(player.get("nationality") or ""),
    ]
    meta = " &nbsp;·&nbsp; ".join(html.escape(part) for part in meta_parts if part)

    photo_url = player.get("image_url")
    photo = (
        f'<img src="{html.escape(photo_url)}" alt="" '
        'style="height:150px; border-radius:12px; margin-bottom:0.9rem;">'
        if isinstance(photo_url, str) and photo_url.startswith("https://")
        else ""
    )

    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.6rem 0 0.4rem 0;">
            {photo}
            <div style="font-size:2.9rem; font-weight:700; line-height:1.15;">
                {html.escape(str(player["player_name"]))}
            </div>
            <div style="font-size:1.1rem; color:{COLOR_MUTED}; padding-top:0.6rem;">
                {meta} &nbsp;·&nbsp; {latest_season} season
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # The app works with the calibrated 50% range: half of the players priced the
    # way the model expects are listed inside it. Without a saved interval model,
    # ±MAPE is the same thing (MAPE is the *median* error), just uncalibrated.
    if "low_50" in player.index and pd.notna(player["low_50"]):
        low, high = float(player["low_50"]), float(player["high_50"])
    else:
        low, high = predicted * (1 - mape), predicted * (1 + mape)

    if pd.isna(score):
        st.info("No market value on record, so there is nothing to compare against.")
    elif market < low:
        st.success(f"**Undervalued.** His listed value is below the model's 50% range "
                   f"({format_eur(low)}–{format_eur(high)}).")
    elif market > high:
        st.warning(f"**Overvalued.** His listed value is above the model's 50% range "
                   f"({format_eur(low)}–{format_eur(high)}).")
    else:
        st.info(f"**Fairly valued.** His listed value sits inside the model's 50% range "
                f"({format_eur(low)}–{format_eur(high)}).")

    if bool(player.get("market_value_imputed", False)):
        st.caption(
            "Transfermarkt had no end-of-season listing for this season, so the market "
            "value shown was interpolated from his neighbouring seasons."
        )

    left, middle, right = st.columns(3)
    left.metric("Model valuation", format_eur(predicted),
                help=f"50% range {format_eur(low)}–{format_eur(high)}")
    middle.metric("Listed market value", format_eur(market))
    right.metric("Undervalue score", "n/a" if pd.isna(score) else f"{score * 100:+.0f}%")

    st.markdown(f"**50% range:** {format_eur(low)}–{format_eur(high)}")
    st.altair_chart(valuation_range_chart(predicted, market, low, high), use_container_width=True)

    st.caption(
        "The blue band is the model's 50% range: for players with this profile, the "
        "model expects half to be listed inside it, a quarter below and a quarter above"
        + (f" — on the test seasons {cover[50]:.0%} fell inside. " if cover else ". ")
        + "Being outside it is common (1 in 2 players), so read it as a lean, not a verdict. "
        "Ranges are wide because a player's price depends on things no stats capture "
        "(contract, agent, hype). The orange dashed line is the value Transfermarkt listed "
        "at the end of the season."
    )

    # ── Value history ───────────────────────────────────────────────────────
    st.subheader("Value over time")
    history = player_history(players, selected_id)
    st.altair_chart(value_history_chart(history), use_container_width=True)
    if len(history) == 1:
        st.caption("Only one season on record for this player.")
    else:
        first, last = history.iloc[0], history.iloc[-1]
        st.caption(
            f"{len(history)} seasons on record, {int(first['year'])}–{int(last['year'])}: "
            f"listed value went from {format_eur(first['market_value'])} to "
            f"{format_eur(last['market_value'])}. Seasons up to {TRAIN_END} were in the "
            "training data, so the blue line there is fitted rather than predicted. "
            "Hollow orange points are interpolated, not real listings."
        )

    # ── Why this valuation ──────────────────────────────────────────────────
    st.subheader("Why this valuation")

    result = explain(model, row)
    feature_values = select_features(row).iloc[0]
    shap_df = pd.DataFrame(
        {
            "feature": list(result["shap_values"].keys()),
            "impact": [float(v) for v in result["shap_values"].values()],
        }
    )
    shap_df["value_label"] = shap_df["feature"].map(
        lambda f: format_feature_value(feature_values.get(f))
    )

    # Season stats, benchmarked against his position — what the valuation sees.
    season_label = f"{latest_season}-{str(latest_season + 1)[-2:]}"
    peers_name = {"DEF": "defenders", "MID": "midfielders", "ATT": "attackers"}.get(
        player["pos_group"], "players in his position")
    st.markdown(f"**{season_label} season stats** — percentile against other {peers_name}")
    stats, n_peers = season_stats(players, player)
    if stats["total"].isna().all():
        st.caption("No stats on record for this season: Understat and FBref only list "
                   "players who appeared.")
    else:
        stats["total"] = stats["total"].map(lambda v: "–" if pd.isna(v) else f"{v:,.1f}".replace(".0", ""))
        stats["per_90"] = stats["per_90"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        st.dataframe(
            stats,
            column_config={
                "stat": "Stat",
                "total": "Season",
                "per_90": "Per 90",
                "percentile": st.column_config.ProgressColumn(
                    "Percentile vs position", min_value=0, max_value=100, format="%d",
                    help="Share of same-position players (≥450 min) he is above. "
                         "Ranked on the per-90 rate where there is one."),
            },
            hide_index=True,
            use_container_width=True,
        )
        st.caption(f"Against {n_peers} {peers_name} with at least 450 minutes this season. "
                   "Attacking stats from Understat; tackles, interceptions, fouls and crosses "
                   "from FBref. Higher percentile means *more*, not necessarily better — "
                   "fouls committed and yellow cards included.")

    st.markdown(
        f"**Full breakdown.** The model starts every player at the average "
        f"valuation of **{format_eur(result['baseline_eur'])}**; each bar below moves "
        f"him up or down from there, and together they land on "
        f"**{format_eur(result['pred_eur'])}**. Hover a bar to see the player's "
        "actual value for that feature."
    )
    st.altair_chart(shap_bar_chart(shap_df), use_container_width=True)
    st.caption(
        "Values are in log-space, where they sum exactly to the prediction -- "
        "individual numbers are not euro amounts. A bar of +0.1 means roughly "
        "+10% on the valuation. Bars at zero are features that did not matter "
        "for this player (e.g. nationality flags that don't apply)."
    )

    with st.expander("What do these features mean?"):
        for feature, definition in FEATURE_GLOSSARY.items():
            st.markdown(f"**{feature}** -- {definition}")

    st.caption(
        "Goalkeepers are not valued: the model is trained on outfield stats only. "
        "Player photos: Transfermarkt."
    )

with leaderboard_tab:
    st.subheader(f"Most undervalued players -- {latest_season} season")

    pos_ctl, min_ctl, count_ctl = st.columns([2, 2, 3])
    position = pos_ctl.selectbox(
        "Position",
        ["All positions", *POSITION_LABELS],
        format_func=lambda key: POSITION_LABELS.get(key, key),
    )
    min_value = min_ctl.select_slider(
        "Minimum market value",
        options=MIN_VALUE_OPTIONS,
        value=1_000_000,
        format_func=format_eur,
    )
    count = count_ctl.slider("How many to show", 5, 50, 20)

    current = players[players["year"] == latest_season]
    table = top_undervalued(
        current,
        position_group=None if position == "All positions" else position,
        limit=count,
        min_market_value=min_value,
    )
    table = table.assign(undervalue_pct=table["undervalue_score"] * 100)
    # money columns in EUR millions: "€10.0M" reads faster than "EUR 10000000"
    money = ["market_value", "pred_value", "low_50", "high_50"]
    table = table.assign(**{c: table[c] / 1e6 for c in money if c in table.columns})

    shown = ["player_name", "club", "age", "pos_group", "market_value", "pred_value",
             "low_50", "high_50", "undervalue_pct"]
    st.dataframe(
        table[[c for c in shown if c in table.columns]],
        column_config={
            "player_name": "Player",
            "club": "Club",
            "age": st.column_config.NumberColumn("Age", format="%.0f"),
            "pos_group": "Position",
            "market_value": st.column_config.NumberColumn("Market value", format="€%.1fM"),
            "pred_value": st.column_config.NumberColumn("Predicted", format="€%.1fM"),
            "low_50": st.column_config.NumberColumn("50% range from", format="€%.1fM"),
            "high_50": st.column_config.NumberColumn("50% range to", format="€%.1fM"),
            "undervalue_pct": st.column_config.NumberColumn("Undervalued by", format="%+.0f%%"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        f"Ranked by (predicted - market) / market across {len(current)} outfield "
        f"players from the {latest_season} season. The minimum market value filter "
        "exists because the score is a ratio: academy players sit at Transfermarkt's "
        "EUR 25K floor, so any sensible prediction reads as thousands of percent "
        "undervalued. Interpolated market values and goalkeepers are excluded."
    )
