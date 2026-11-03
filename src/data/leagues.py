"""The leagues Transfer Edge covers, and how each source names them.

Keyed by Transfermarkt's competition code, which is what the panel's `league`
column stores. Every scraper reads its league names and URLs from here, so
adding a league means adding one entry, not editing three notebooks.

Values are always EUR: Transfermarkt's market-value API returns euros for every
league, Premier League included (only the .co.uk website renders pounds). The
scrapers store the API's `currency` field and refuse anything else.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class League:
    code: str            # Transfermarkt competition code, e.g. "GB1"
    name: str            # display name
    country: str         # the league's country (not the players' nationality)
    tm_slug: str         # Transfermarkt URL slug
    understat: str       # Understat `league` parameter
    fbref_id: int        # FBref competition id
    fbref_name: str      # FBref URL suffix
    clubs: int           # clubs per season
    clubs_changes: tuple = ()   # ((first_year, clubs), ...) when the league resized


LEAGUES = {
    lg.code: lg
    for lg in (
        League("ES1", "LaLiga", "Spain", "laliga", "La_liga", 12, "La-Liga", 20),
        League("GB1", "Premier League", "England", "premier-league", "EPL", 9, "Premier-League", 20),
        League("L1", "Bundesliga", "Germany", "bundesliga", "Bundesliga", 20, "Bundesliga", 18),
        League("IT1", "Serie A", "Italy", "serie-a", "Serie_A", 11, "Serie-A", 20),
        # Ligue 1 went from 20 clubs to 18 in 2023-24.
        League("FR1", "Ligue 1", "France", "ligue-1", "Ligue_1", 13, "Ligue-1", 20, ((2023, 18),)),
    )
}


def clubs_in_season(code: str, year: int) -> int:
    """Clubs in `code` in the season starting `year` (2023 == 2023-24)."""
    league = LEAGUES[code]
    clubs = league.clubs
    for first_year, n in league.clubs_changes:
        if year >= first_year:
            clubs = n
    return clubs


def tm_league_url(code: str, year: int) -> str:
    league = LEAGUES[code]
    return (f"https://www.transfermarkt.us/{league.tm_slug}/startseite/wettbewerb/"
            f"{code}/plus/?saison_id={year}")


def fbref_url(code: str, year: int, stat: str = "misc") -> str:
    league = LEAGUES[code]
    season = f"{year}-{year + 1}"
    return (f"https://fbref.com/en/comps/{league.fbref_id}/{season}/{stat}/"
            f"{season}-{league.fbref_name}-Stats")
