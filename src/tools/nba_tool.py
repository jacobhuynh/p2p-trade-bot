"""
src/tools/nba_tool.py

Lightweight nba_api wrapper for the Quant Agent.

Parses team abbreviations from a KXNBAGAME Kalshi ticker and fetches
each team's recent W/L record via the NBA Stats API.

This is supplementary context for the quant LLM summary — the core
calibration gap analysis does NOT depend on it.  All calls are wrapped
with a timeout and return None on failure so the pipeline never blocks.

Public API
----------
get_team_recent_records(ticker, last_n=10) -> dict | None
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

_NBA_API_TIMEOUT = 15  # seconds — nba.com cold-start can be slow; 5s was too low


# ── Kalshi abbreviation → NBA Stats API team name fragment ───────────────────
# nba_api identifies teams by full name or team_id.
# This mapping translates Kalshi 3-letter codes to the full team name used
# by nba_api.stats.static.teams for lookup.

_KALSHI_TO_NBA_NAME: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


# ── Ticker parser ─────────────────────────────────────────────────────────────

def _parse_teams_from_ticker(ticker: str) -> Optional[tuple[str, str]]:
    """
    Extract (home_abbr, away_abbr) from a KXNBAGAME Kalshi ticker.

    Format: KXNBAGAME-{YYMONDD}{HOME}{AWAY}-{SIDE}
    E.g.    KXNBAGAME-25JAN15LACBOS-NO  →  ("LAC", "BOS")

    Returns None if the ticker is not a KXNBAGAME ticker or can't be parsed.
    """
    if not ticker.startswith("KXNBAGAME"):
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    middle = parts[1]  # e.g. "25JAN15LACBOS"

    date_match = re.match(r"^\d{2}[A-Z]{3}\d{2}", middle)
    if not date_match:
        return None
    remainder = middle[date_match.end():]  # e.g. "LACBOS"

    # Must use {2,3} — {2,4} is greedy and merges e.g. "LACBOS" into ("LACB","OS").
    tokens = re.findall(r"[A-Z]{2,3}", remainder)
    if len(tokens) < 2:
        return None
    return tokens[0], tokens[1]


def _get_team_id(abbr: str) -> Optional[int]:
    """Look up nba_api team ID from a Kalshi abbreviation."""
    try:
        from nba_api.stats.static import teams as nba_teams
        full_name = _KALSHI_TO_NBA_NAME.get(abbr.upper())
        if not full_name:
            return None
        results = nba_teams.find_teams_by_full_name(full_name)
        if results:
            return results[0]["id"]
        return None
    except Exception:
        return None


def _fetch_recent_record(team_id: int, last_n: int) -> Optional[dict]:
    """
    Pull the last `last_n` games for a team from LeagueGameFinder and
    compute wins, losses, home record, and away record.

    Returns None on timeout or any other error.
    """
    try:
        from nba_api.stats.endpoints import LeagueGameFinder
        finder = LeagueGameFinder(
            team_id_nullable=team_id,
            timeout=_NBA_API_TIMEOUT,
        )
        df = finder.get_data_frames()[0]
        if df is None or df.empty:
            return None

        recent = df.head(last_n)
        wins  = int((recent["WL"] == "W").sum())
        total = len(recent)

        home_games = recent[recent["MATCHUP"].str.contains("vs\\.")]
        away_games = recent[recent["MATCHUP"].str.contains("@")]
        home_wins  = int((home_games["WL"] == "W").sum())
        away_wins  = int((away_games["WL"] == "W").sum())

        return {
            f"last{last_n}": f"{wins}-{total - wins}",
            "home_record":   f"{home_wins}-{len(home_games) - home_wins}",
            "away_record":   f"{away_wins}-{len(away_games) - away_wins}",
        }
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_team_recent_records(ticker: str, last_n: int = 10) -> Optional[dict]:
    """
    Return recent W/L records for the two teams in a KXNBAGAME ticker.

    Returns:
        {
          "home": {"abbr": "LAC", "last10": "6-4", "home_record": "4-1", "away_record": "2-3"},
          "away": {"abbr": "BOS", "last10": "8-2", "home_record": "5-0", "away_record": "3-2"},
        }
        or None if the ticker is not KXNBAGAME, teams can't be parsed, or nba_api fails.

    All failures return None gracefully — the quant pipeline continues without this data.
    """
    parsed = _parse_teams_from_ticker(ticker)
    if parsed is None:
        return None
    home_abbr, away_abbr = parsed

    home_id = _get_team_id(home_abbr)
    away_id = _get_team_id(away_abbr)

    if home_id is None or away_id is None:
        return None

    # Fetch both teams in parallel — cuts total time from sum to max of the two calls.
    # Each call can take 5–15s on cold start; sequential would double the wait.
    home_record: Optional[dict] = None
    away_record: Optional[dict] = None

    with ThreadPoolExecutor(max_workers=2) as ex:
        future_home = ex.submit(_fetch_recent_record, home_id, last_n)
        future_away = ex.submit(_fetch_recent_record, away_id, last_n)
        home_record = future_home.result()
        away_record = future_away.result()

    if home_record is None and away_record is None:
        return None

    return {
        "home": {"abbr": home_abbr, **(home_record or {})},
        "away": {"abbr": away_abbr, **(away_record or {})},
    }
