"""FBref season stats keyed by the Transfermarkt panel's (year, player_id).

FBref has no Transfermarkt id. The players are linked by the deterministic
crosswalk (src/data/crosswalk.py: the public FBref -> Transfermarkt map first,
then the name and club rules), and this module only applies those links.

    fbref_stats.parquet (notebook 05)  ->  match_to_panel()  ->  fbref_matched.parquet
    one row per (season, league, player, club)                 one row per (season, player_id)
"""

import pandas as pd

# Season counts kept from FBref. Everything else it still publishes (cards,
# offsides, penalties) is either noise for valuation or already in the panel.
COUNTS = ["minutes_90s", "tackles_won", "interceptions", "fouls", "fouled", "crosses"]


def aggregate_seasons(fbref: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, FBref player): every spell of the season summed.

    A mid-season move is one row per club on FBref, and a move between two of
    the leagues one row per league. The panel keeps one row per player-season
    and Understat's numbers are whole-season too, so FBref is summed to match.
    """
    counts = [c for c in COUNTS if c in fbref.columns]
    return (
        fbref.groupby(["year", "fbref_id"], as_index=False)
        .agg(player_name_fb=("player_name_fb", "first"),
             birth_year=("birth_year", "first"),
             teams_fb=("team", lambda t: ", ".join(sorted(set(t)))),
             fb_leagues=("league_code", lambda s: ",".join(sorted(set(s)))),
             **{c: (c, "sum") for c in counts})
    )


def match_to_panel(fbref: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """FBref season stats keyed by (year, player_id), through the crosswalk links.

    `links` is the crosswalk's FBref part (src_id = FBref id, player_id, rule).
    Returns year, player_id, fbref_id, fbref_rule, fb_leagues and the COUNTS
    prefixed `fb_` (e.g. fb_tackles_won) so they can't collide with Understat columns.
    """
    fb = aggregate_seasons(fbref)
    links = links.loc[links["source"] == "fbref", ["src_id", "player_id", "rule"]]
    out = fb.merge(links, left_on="fbref_id", right_on="src_id")
    counts = [c for c in COUNTS if c in out.columns]
    out = out.rename(columns={"rule": "fbref_rule", **{c: f"fb_{c}" for c in counts}})
    return out[["year", "player_id", "fbref_id", "fbref_rule", "fb_leagues",
                *[f"fb_{c}" for c in counts]]].reset_index(drop=True)
