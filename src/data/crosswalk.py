"""Deterministic player links: Transfermarkt id <-> Understat id / FBref id.

Replaces the per-season fuzzy joins. Players are linked, not rows: a link found
in any season holds for every season (both sides' ids are stable), and every
link records the rule that made it, so each one can be traced.

Rules, in priority order. All of them compare players within the same season and
league, and are one-to-one per season:

    override      data/manual/id_overrides.csv, written by a person
    public_map    FBref only: worldfootballR's FBref -> Transfermarkt URL map
    name_club     same normalised name, same club
    subset_club   same club, one name's words all inside the other's
                  ("Kepa" / "Kepa Arrizabalaga"), unique both ways in the club
    initial_club  same club, same surname and first initial, unique both ways
    name          same normalised name, unique in the league-season on both sides,
                  clubs differing (a loan or move the rosters disagree on). Last:
                  run earlier, it paired Levante's "Toño" with Real Sociedad's.

When both sides know the birth year (FBref does, Understat doesn't), it must
agree, for every name rule. The public map links by profile URL, not by name, so
there the birth year only guards against gross errors: FBref's is off by a year
or two for a handful of players (Dwight Gayle, born 1989, is 1990 there). A player linked to two different ids in different seasons
keeps the link seen in more seasons; the loser goes to the review file.

Clubs are matched the same way, without text similarity: a Transfermarkt club and
a source team are the same club when their squads share the most players with an
identical normalised name. The table is written to data/manual/ for checking.

Leftover source players (they played, so they should exist on Transfermarkt) go
to data/manual/review_<source>.csv with the closest Transfermarkt name in the same
club as a hint. Hints are never applied; copy a row into id_overrides.csv to do so.
"""

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = Path(__file__).parents[2] / "data"
MANUAL_DIR = DATA_DIR / "manual"
OVERRIDES_PATH = MANUAL_DIR / "id_overrides.csv"
CROSSWALK_PATH = DATA_DIR / "id_crosswalk.parquet"
CLUBS_PATH = MANUAL_DIR / "club_crosswalk.csv"
FBREF_MAP_URL = ("https://raw.githubusercontent.com/JaseZiv/worldfootballR_data/master/"
                 "raw-data/fbref-tm-player-mapping/output/fbref_to_tm_mapping.csv")
FBREF_MAP_PATH = DATA_DIR / "_fbref" / "fbref_to_tm_mapping.csv"

RULES = ["override", "public_map", "name_club", "subset_club", "initial_club", "name"]
MIN_SHARED_PLAYERS = 3     # a club pairing backed by fewer exact names is not trusted
SEED_BIRTH_TOLERANCE = 2   # years a public-map link's two birth years may differ

# Letters NFKD does not decompose into a base letter + accent.
_LETTERS = str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ß": "ss", "ł": "l", "Ł": "l",
                          "đ": "d", "Đ": "d", "ı": "i", "œ": "oe", "Œ": "oe", "þ": "th", "ð": "d"})


def normalize_name(name) -> str:
    """'Martin Ødegaard' -> 'martin odegaard'; 'Marc-André ter Stegen' -> 'marc andre ter stegen'."""
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", html.unescape(name).translate(_LETTERS))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"['’]", "", text)             # N'Zonzi -> nzonzi
    text = re.sub(r"[-.]", " ", text)             # Marc-André -> marc andre
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text)).strip()


# ── Clubs ───────────────────────────────────────────────────────────────────

def club_table(tm: pd.DataFrame, src: pd.DataFrame, source: str) -> pd.DataFrame:
    """Pair each Transfermarkt club with a source team, per league, by shared exact names.

    `tm` has year, league_code, club, name_norm; `src` has year, league_code,
    team (one team per row), name_norm. Greedy one-to-one by shared count, except
    that a team may take a second Transfermarkt name for seasons the first never
    covers: a club renamed on one side only ("SPAL" -> "SPAL 2013").
    """
    pairs = tm.merge(src, on=["year", "league_code", "name_norm"])
    counts = (pairs.groupby(["league_code", "club", "team"]).size()
              .rename("shared_players").reset_index()
              .sort_values(["shared_players", "club", "team"], ascending=[False, True, True]))
    seasons = tm.groupby(["league_code", "club"])["year"].agg(set).to_dict()
    used_club, team_seasons, rows = set(), {}, []
    for r in counts.itertuples(index=False):
        if r.shared_players < MIN_SHARED_PLAYERS:
            break
        club, team = (r.league_code, r.club), (r.league_code, r.team)
        if club in used_club or team_seasons.get(team, set()) & seasons[club]:
            continue
        used_club.add(club)
        team_seasons[team] = team_seasons.get(team, set()) | seasons[club]
        rows.append(r._asdict())
    table = pd.DataFrame(rows, columns=["league_code", "club", "team", "shared_players"])
    table.insert(0, "source", source)
    return table.sort_values(["league_code", "club"]).reset_index(drop=True)


def check_clubs(tm: pd.DataFrame, table: pd.DataFrame, source: str) -> list[tuple]:
    """(league_code, club) pairs in `tm` with no source team. Empty means every club is mapped."""
    have = set(zip(table["league_code"], table["club"]))
    return sorted({(lg, c) for lg, c in zip(tm["league_code"], tm["club"])} - have)


# ── Players ─────────────────────────────────────────────────────────────────

def _born_ok(t_born, s_born) -> bool:
    return pd.isna(t_born) or pd.isna(s_born) or int(t_born) == int(s_born)


def _season_links(tm: pd.DataFrame, src: pd.DataFrame) -> list[tuple]:
    """Links within one (season, league): [(player_id, src_id, rule)], one-to-one.

    `tm`: player_id, club_team (the club's source-side name), name_norm, birth_year.
    `src`: src_id, teams (frozenset of team names), name_norm, birth_year.
    """
    t_rows = tm.drop_duplicates(["player_id", "club_team"]).to_dict("records")
    s_rows = src.to_dict("records")
    used_t, used_s, links = set(), set(), []

    def accept(cands, rule):
        # cands: (player_id, src_id) pairs, already unique-filtered by the rule
        for pid, sid in sorted(cands):
            if pid not in used_t and sid not in used_s:
                used_t.add(pid)
                used_s.add(sid)
                links.append((pid, sid, rule))

    def free():
        return ([t for t in t_rows if t["player_id"] not in used_t],
                [s for s in s_rows if s["src_id"] not in used_s])

    def same_club(t, s):
        return t["club_team"] is not None and t["club_team"] in s["teams"]

    def unique_pairs(match, block):
        """Pairs where each side has exactly one candidate among the free players.

        `block` is "name" or "club": only source players sharing the name, or
        the club, are compared — every rule requires one or the other.
        """
        ts, ss = free()
        index = {}
        for s in ss:
            for key in ([s["name_norm"]] if block == "name" else s["teams"]):
                index.setdefault(key, []).append(s)
        cands = set()
        for t in ts:
            for s in index.get(t["name_norm"] if block == "name" else t["club_team"], []):
                if _born_ok(t["birth_year"], s["birth_year"]) and match(t, s):
                    cands.add((t["player_id"], s["src_id"]))
        n_t, n_s = {}, {}
        for p, s in cands:
            n_t[p] = n_t.get(p, 0) + 1
            n_s[s] = n_s.get(s, 0) + 1
        return {(p, s) for p, s in cands if n_t[p] == 1 and n_s[s] == 1}

    def subset(t, s):
        a, b = set(t["name_norm"].split()), set(s["name_norm"].split())
        return bool(a) and bool(b) and (a <= b or b <= a)

    def initial(t, s):
        a, b = t["name_norm"].split(), s["name_norm"].split()
        return bool(a) and bool(b) and a[-1] == b[-1] and a[0][0] == b[0][0]

    accept(unique_pairs(same_club, "name"), "name_club")
    accept(unique_pairs(subset, "club"), "subset_club")
    accept(unique_pairs(initial, "club"), "initial_club")
    accept(unique_pairs(lambda t, s: True, "name"), "name")
    return links


def link_players(tm: pd.DataFrame, src: pd.DataFrame, clubs: pd.DataFrame, source: str,
                 overrides: pd.DataFrame | None = None, seed: pd.DataFrame | None = None):
    """Player-level links for one source.

    tm:   year, league_code, club, player_id, player_name, birth_year
    src:  year, league_code, src_id, src_name, teams (frozenset), birth_year
    seed: optional pre-made links (src_id, player_id, rule), e.g. the public map;
          applied after overrides, before the rules.

    Returns (links, conflicts): links has source, src_id, player_id, rule,
    n_seasons; conflicts lists the links dropped because an id was claimed twice.
    """
    tm = tm.assign(name_norm=tm["player_name"].map(normalize_name))
    src = src.assign(name_norm=src["src_name"].map(normalize_name))
    team_of = dict(zip(zip(clubs["league_code"], clubs["club"]), clubs["team"]))
    tm["club_team"] = [team_of.get(k) for k in zip(tm["league_code"], tm["club"])]

    fixed = []   # (src_id, player_id, rule) decided before the season rules
    for frame, rule in ((overrides, "override"), (seed, None)):
        if frame is None or frame.empty:
            continue
        taken_s = {s for s, _, _ in fixed}
        taken_p = {p for _, p, _ in fixed}
        for r in frame.itertuples(index=False):
            if r.src_id in taken_s or (isinstance(r.player_id, str) and r.player_id in taken_p):
                continue
            fixed.append((r.src_id, r.player_id, rule or r.rule))
    fixed_s = {s for s, _, _ in fixed}
    fixed_p = {p for _, p, _ in fixed if isinstance(p, str)}

    season_links = []
    for (year, league), t in tm.groupby(["year", "league_code"]):
        s = src[(src["year"] == year) & (src["league_code"] == league)]
        t = t[~t["player_id"].isin(fixed_p)]
        s = s[~s["src_id"].isin(fixed_s)]
        if len(t) and len(s):
            season_links += [(year, *link) for link in _season_links(t, s)]

    found = pd.DataFrame(season_links, columns=["year", "player_id", "src_id", "rule"])
    found["rank"] = found["rule"].map(RULES.index)
    per_pair = (found.groupby(["src_id", "player_id"])
                .agg(n_seasons=("year", "nunique"), rank=("rank", "min")).reset_index()
                .sort_values(["n_seasons", "rank", "src_id", "player_id"],
                             ascending=[False, True, True, True]))
    used_s, used_p, keep, conflicts = set(fixed_s), set(fixed_p), [], []
    for r in per_pair.itertuples(index=False):
        if r.src_id in used_s or r.player_id in used_p:
            conflicts.append(r._asdict())
            continue
        used_s.add(r.src_id)
        used_p.add(r.player_id)
        keep.append({"src_id": r.src_id, "player_id": r.player_id,
                     "rule": RULES[r.rank], "n_seasons": r.n_seasons})

    links = pd.DataFrame(keep + [{"src_id": s, "player_id": p, "rule": rule, "n_seasons": None}
                                 for s, p, rule in fixed if isinstance(p, str)])
    links.insert(0, "source", source)
    return links, pd.DataFrame(conflicts)


def review_table(tm: pd.DataFrame, src: pd.DataFrame, links: pd.DataFrame, clubs: pd.DataFrame,
                 source: str, minutes_col: str, never: set | None = None) -> pd.DataFrame:
    """Unlinked source players who played, with the closest same-club TM name as a hint.

    `never`: ids an override says never to link; they were reviewed already.
    """
    done = set(links["src_id"]) | (never or set())
    left = src[~src["src_id"].isin(done) & (src[minutes_col].fillna(0) > 0)]
    if left.empty:
        return pd.DataFrame()
    team_of = dict(zip(zip(clubs["league_code"], clubs["club"]), clubs["team"]))
    tm = tm.assign(club_team=[team_of.get(k) for k in zip(tm["league_code"], tm["club"])],
                   name_norm=tm["player_name"].map(normalize_name))
    rows = []
    for r in left.itertuples(index=False):
        pool = tm[(tm["year"] == r.year) & (tm["league_code"] == r.league_code)
                  & tm["club_team"].isin(r.teams)]
        hint = (process.extractOne(normalize_name(r.src_name), pool["name_norm"].tolist(),
                                   scorer=fuzz.token_sort_ratio) if len(pool) else None)
        hint_row = pool.iloc[hint[2]] if hint else None
        rows.append({
            "source": source, "src_id": r.src_id, "src_name": r.src_name, "year": r.year,
            "league_code": r.league_code, "teams": ", ".join(sorted(r.teams)),
            "minutes": getattr(r, minutes_col),
            "hint_player_id": hint_row["player_id"] if hint else None,
            "hint_player_name": hint_row["player_name"] if hint else None,
            "hint_score": round(hint[1]) if hint else None,
        })
    return (pd.DataFrame(rows).sort_values(["src_id", "year"])
            .drop_duplicates("src_id", keep="last").sort_values("minutes", ascending=False))


# ── Inputs ─────────────────────────────────────────────────────────────────

def load_overrides(source: str) -> pd.DataFrame:
    """Rows of id_overrides.csv for `source`. An empty player_id means "never link this id"."""
    if not OVERRIDES_PATH.exists():
        return pd.DataFrame(columns=["src_id", "player_id"])
    o = pd.read_csv(OVERRIDES_PATH, dtype=str)
    o = o[o["source"] == source]
    return o.assign(player_id=o["player_id"].where(o["player_id"].notna(), None))[["src_id", "player_id"]]


def fbref_public_map(download: bool = False) -> pd.DataFrame:
    """worldfootballR's map as (src_id=fbref_id, player_id). Cached in data/_fbref/.

    Only downloaded when asked, so the same cached file gives the same links on
    every run; refresh it deliberately once a season.
    """
    if download or not FBREF_MAP_PATH.exists():
        FBREF_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(FBREF_MAP_URL, encoding="latin-1").to_csv(FBREF_MAP_PATH, index=False)
    m = pd.read_csv(FBREF_MAP_PATH, dtype=str)
    m = pd.DataFrame({
        "src_id": m["UrlFBref"].str.extract(r"/players/([0-9a-f]{8})/")[0],
        "player_id": m["UrlTmarkt"].str.extract(r"/spieler/(\d+)")[0],
    }).dropna()
    # An id that appears twice on either side is ambiguous: use neither.
    m = m[~m["src_id"].duplicated(keep=False) & ~m["player_id"].duplicated(keep=False)]
    return m.assign(rule="public_map")


def tm_players(panel: pd.DataFrame) -> pd.DataFrame:
    """Transfermarkt side: one row per (season, league, club, player)."""
    tm = panel[["year", "league_code", "club", "player_id", "player_name", "birth_date"]].copy()
    tm["year"] = tm["year"].astype(int)
    tm["player_id"] = tm["player_id"].astype(str)
    tm["birth_year"] = pd.to_datetime(tm["birth_date"], errors="coerce").dt.year
    return tm.drop(columns="birth_date").drop_duplicates()


def understat_players(us: pd.DataFrame) -> pd.DataFrame:
    """Understat side. A mid-season move inside the league is one row, teams 'A,B'."""
    return pd.DataFrame({
        "year": us["year"].astype(int), "league_code": us["league_code"],
        "src_id": us["understat_id"].astype(str), "src_name": us["player_name_us"],
        "teams": us["team_us"].map(lambda t: frozenset(x.strip() for x in str(t).split(","))),
        "birth_year": float("nan"), "minutes": us["minutes"],
    })


def fbref_players(fb: pd.DataFrame) -> pd.DataFrame:
    """FBref side: one row per (season, league, player), a mid-season move's clubs joined."""
    fb = fb.assign(year=fb["year"].astype(int), fbref_id=fb["fbref_id"].astype(str))
    g = fb.groupby(["year", "league_code", "fbref_id"])
    return pd.DataFrame({
        "src_name": g["player_name_fb"].first(),
        "teams": g["team"].agg(frozenset),
        "birth_year": pd.to_numeric(g["birth_year"].first(), errors="coerce"),
        "minutes": g["minutes_90s"].sum(min_count=1),
    }).reset_index().rename(columns={"fbref_id": "src_id"})


def _replace_rows(path: Path, new: pd.DataFrame, sources: list[str]) -> None:
    """Write `new` to `path`, keeping the rows of sources not rebuilt this time."""
    read, write = ((pd.read_parquet, pd.DataFrame.to_parquet) if path.suffix == ".parquet"
                   else (pd.read_csv, pd.DataFrame.to_csv))
    if path.exists():
        old = read(path, dtype=str) if path.suffix == ".csv" else read(path)
        new = pd.concat([old[~old["source"].isin(sources)], new], ignore_index=True)
    write(new.sort_values("source", kind="stable"), path, index=False)


def build_crosswalk(panel: pd.DataFrame, understat: pd.DataFrame | None,
                    fbref: pd.DataFrame | None, write: bool = True) -> dict:
    """Clubs, links, conflicts and review tables for the sources given; optionally written.

    Every input needs `league_code`. Pass None for a source to leave its links
    as they are on disk. Returns a dict with `crosswalk` (this call's links),
    plus per source: clubs_<s>, conflicts_<s>, review_<s>, unmapped_clubs_<s>.
    """
    tm = tm_players(panel)
    out, all_links, sources = {}, [], []
    if understat is not None:
        sources.append(("understat", understat_players(understat), None))
    if fbref is not None:
        sources.append(("fbref", fbref_players(fbref), fbref_public_map()))
    for source, src, seed in sources:
        exploded = src.explode("teams").rename(columns={"teams": "team"})
        tm_norm = tm.assign(name_norm=tm["player_name"].map(normalize_name))
        src_norm = exploded.assign(name_norm=exploded["src_name"].map(normalize_name))
        clubs = club_table(tm_norm, src_norm, source)
        if seed is not None:
            # Keep only map entries for players present on both sides, with close birth years.
            born_t = tm.drop_duplicates("player_id").set_index("player_id")["birth_year"]
            born_s = src.drop_duplicates("src_id").set_index("src_id")["birth_year"]
            seed = seed[seed["player_id"].isin(born_t.index) & seed["src_id"].isin(born_s.index)]
            ok = [pd.isna(born_t[p]) or pd.isna(born_s[s]) or abs(born_t[p] - born_s[s]) <= SEED_BIRTH_TOLERANCE
                  for s, p in zip(seed["src_id"], seed["player_id"])]
            seed = seed[ok]
        overrides = load_overrides(source)
        links, conflicts = link_players(tm, src, clubs, source, overrides, seed)
        out[f"clubs_{source}"] = clubs
        out[f"unmapped_clubs_{source}"] = check_clubs(tm, clubs, source)
        out[f"conflicts_{source}"] = conflicts
        never = set(overrides.loc[overrides["player_id"].isna(), "src_id"])
        out[f"review_{source}"] = review_table(tm, src, links, clubs, source, "minutes", never)
        all_links.append(links)
    out["crosswalk"] = pd.concat(all_links, ignore_index=True)

    if write:
        # Notebook 02 builds the Understat links, notebook 05 the FBref ones:
        # each write replaces its own sources' rows and keeps the others'.
        built = [s for s, _, _ in sources]
        MANUAL_DIR.mkdir(parents=True, exist_ok=True)
        _replace_rows(CROSSWALK_PATH, out["crosswalk"], built)
        _replace_rows(CLUBS_PATH, pd.concat([out[f"clubs_{s}"] for s in built]), built)
        for s, _, _ in sources:
            out[f"review_{s}"].to_csv(MANUAL_DIR / f"review_{s}.csv", index=False)
        if not OVERRIDES_PATH.exists():
            pd.DataFrame(columns=["source", "src_id", "player_id", "note"]).to_csv(OVERRIDES_PATH, index=False)
    return out
