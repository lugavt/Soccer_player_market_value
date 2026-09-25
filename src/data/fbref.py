"""Match FBref season stats to the Transfermarkt panel.

FBref has no Transfermarkt ID, so rows are matched per season by name, like the
Understat join in notebook 02 — with one extra key both sides carry: the birth
year. When both rows have one, it must be equal, which separates namesakes far
more reliably than the club does.

    fbref_stats.parquet (notebook 05)  ->  match_to_panel()  ->  fbref_matched.parquet
    one row per (season, player, club)                          one row per (season, player_id)
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Season counts kept from FBref. Everything else it still publishes (cards,
# offsides, penalties) is either noise for valuation or already in the panel.
COUNTS = ["minutes_90s", "tackles_won", "interceptions", "fouls", "fouled", "crosses"]

NAME_THRESHOLD = 85        # same bar as the Understat join when birth year can't help
NAME_THRESHOLD_BORN = 70   # birth year + season + league already narrow it to a handful
TEAM_FLOOR = 60            # club must corroborate an inexact name without a birth year


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", _strip_accents(name).lower().strip()))


def normalize_team(team) -> str:
    """Punctuation becomes a space so a list of clubs stays separate tokens."""
    if not isinstance(team, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", _strip_accents(team).lower())).strip()


def aggregate_seasons(fbref: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, FBref player): a mid-season transfer's spells summed.

    The Transfermarkt panel keeps one row per player-season, and Understat's
    numbers are full-season too, so FBref has to match that grain.
    """
    counts = [c for c in COUNTS if c in fbref.columns]
    return (
        fbref.groupby(["year", "fbref_id"], as_index=False)
        .agg(player_name_fb=("player_name_fb", "first"),
             birth_year=("birth_year", "first"),
             teams_fb=("team", lambda t: ", ".join(sorted(set(t)))),
             **{c: (c, "sum") for c in counts})
    )


def _match_season(tm: pd.DataFrame, fb: pd.DataFrame) -> list[tuple]:
    """Greedy one-to-one assignment within one season: (tm_row, fb_row, name_score)."""
    names = process.cdist(tm["name_norm"], fb["name_norm"], scorer=fuzz.token_sort_ratio,
                          dtype=np.float32, workers=-1)
    teams = process.cdist(tm["team_norm"], fb["team_norm"], scorer=fuzz.token_set_ratio,
                          dtype=np.float32, workers=-1)

    tm_born = tm["birth_year"].to_numpy(dtype=float)[:, None]
    fb_born = fb["birth_year"].to_numpy(dtype=float)[None, :]
    both_known = ~np.isnan(tm_born) & ~np.isnan(fb_born)
    same_born = both_known & (tm_born == fb_born)

    candidate = np.where(
        both_known,
        same_born & (names >= NAME_THRESHOLD_BORN),                      # birth year is the gate
        (names >= NAME_THRESHOLD) & ((names >= 100) | (teams >= TEAM_FLOOR)),
    )
    ti, fi = np.where(candidate)
    order = np.lexsort((-teams[ti, fi], -names[ti, fi]))   # name first, club breaks ties

    used_tm, used_fb, pairs = set(), set(), []
    for k in order:
        t, f = int(ti[k]), int(fi[k])
        if t in used_tm or f in used_fb:
            continue
        used_tm.add(t)
        used_fb.add(f)
        pairs.append((t, f, float(names[t, f])))
    return pairs


def match_to_panel(panel: pd.DataFrame, fbref: pd.DataFrame) -> pd.DataFrame:
    """FBref season stats keyed by the panel's (year, player_id).

    Returns only matched rows: year, player_id, fbref_id, fbref_match_score and
    the COUNTS prefixed `fb_` (e.g. fb_tackles_won) so they can't collide with
    Understat columns.
    """
    fb = aggregate_seasons(fbref)
    fb["name_norm"] = fb["player_name_fb"].map(normalize_name)
    fb["team_norm"] = fb["teams_fb"].map(normalize_team)

    tm = panel[["year", "player_id", "player_name", "club", "birth_date"]].copy()
    tm["year"] = tm["year"].astype(int)
    tm["name_norm"] = tm["player_name"].map(normalize_name)
    tm["team_norm"] = tm["club"].map(normalize_team)
    tm["birth_year"] = pd.to_datetime(tm["birth_date"], errors="coerce").dt.year

    out = []
    for year in sorted(set(tm["year"]) & set(fb["year"])):
        t = tm[tm["year"] == year].reset_index(drop=True)
        f = fb[fb["year"] == year].reset_index(drop=True)
        for ti, fi, score in _match_season(t, f):
            row = {"year": year, "player_id": t.at[ti, "player_id"],
                   "fbref_id": f.at[fi, "fbref_id"], "fbref_match_score": score}
            row.update({f"fb_{c}": f.at[fi, c] for c in COUNTS if c in f.columns})
            out.append(row)
    return pd.DataFrame(out)
