"""
src/tools/espn_tool.py

Lightweight wrapper around the ESPN hidden NBA scoreboard API.

Used by two consumers:
  - src/agents/quant.py  : live game context (injuries, game status) for the
                           qualitative LLM summary
  - src/settle.py        : resolution check — is the game STATUS_FINAL?

Public API
----------
get_nba_scoreboard(date=None)  -> list[dict]
find_game(ticker, search_days=2) -> dict | None
"""

import re
import requests
from datetime import datetime, timedelta, timezone

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)

# Kalshi 3-letter abbreviation → ESPN abbreviation (only where they differ)
_KALSHI_TO_ESPN: dict[str, str] = {
    "GSW": "GS",   # Golden State Warriors
    "NOP": "NO",   # New Orleans Pelicans
    "SAS": "SA",   # San Antonio Spurs
    "OKC": "OKC",  # Oklahoma City Thunder (same)
    "UTA": "UTAH", # Utah Jazz
    "PHX": "PHX",  # Phoenix Suns (same)
    "POR": "POR",  # Portland Trail Blazers (same)
    "MEM": "MEM",  # Memphis Grizzlies (same)
    "MIN": "MIN",  # Minnesota Timberwolves (same)
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_espn_abbr(kalshi_abbr: str) -> str:
    """Convert a Kalshi team abbreviation to an ESPN abbreviation."""
    return _KALSHI_TO_ESPN.get(kalshi_abbr.upper(), kalshi_abbr.upper())


def _parse_ticker(ticker: str) -> tuple[str, str] | None:
    """
    Extract (home_abbr, away_abbr) from a KXNBAGAME ticker.

    Ticker format:  KXNBAGAME-{YYMONDD}{HOME}{AWAY}-{SIDE}
    Example:        KXNBAGAME-25JAN15LACBOS-NO  →  (LAC, BOS)

    The date portion is 7 chars (YYMONDD e.g. "25JAN15").
    Team abbreviations are 2–3 uppercase letters immediately after.
    We use a greedy regex that matches two consecutive uppercase sequences.
    """
    if not ticker.startswith("KXNBAGAME"):
        return None
    parts = ticker.split("-")
    if len(parts) < 2:
        return None
    middle = parts[1]  # e.g. "25JAN15LACBOS"

    # Strip leading date: digits + month letters (YYMONDD pattern)
    # e.g. "25JAN15" → 7 chars
    date_match = re.match(r"^\d{2}[A-Z]{3}\d{2}", middle)
    if not date_match:
        return None
    remainder = middle[date_match.end():]  # e.g. "LACBOS"

    # Split remainder into two uppercase tokens (2-3 chars each).
    # Must use {2,3} — {2,4} is greedy and merges e.g. "LACBOS" into ("LACB","OS").
    tokens = re.findall(r"[A-Z]{2,3}", remainder)
    if len(tokens) < 2:
        return None
    return tokens[0], tokens[1]


def _date_strings(search_days: int) -> list[str]:
    """Return last search_days date strings in YYYYMMDD format (today first)."""
    today = datetime.now(timezone.utc).date()
    return [(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(search_days)]


# ── Public API ────────────────────────────────────────────────────────────────

def get_nba_scoreboard(date: str | None = None) -> list[dict]:
    """
    Fetch NBA games for a given date (YYYYMMDD) or today if None.

    Returns a list of game dicts:
      {
        home_abbr, away_abbr, home_name, away_name,
        status,          # e.g. "STATUS_SCHEDULED", "STATUS_IN_PROGRESS", "STATUS_FINAL"
        home_score,      # int or None
        away_score,      # int or None
        winner_abbr,     # str or None (only set when STATUS_FINAL)
        game_date,       # YYYYMMDD string
      }
    """
    params: dict = {}
    if date:
        params["dates"] = date

    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    games: list[dict] = []
    for event in data.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors", [])

        home: dict = {}
        away: dict = {}
        for c in competitors:
            abbr = c.get("team", {}).get("abbreviation", "")
            name = c.get("team", {}).get("displayName", "")
            score = c.get("score")
            try:
                score_int = int(score) if score is not None else None
            except (ValueError, TypeError):
                score_int = None
            if c.get("homeAway") == "home":
                home = {"abbr": abbr, "name": name, "score": score_int}
            else:
                away = {"abbr": abbr, "name": name, "score": score_int}

        status_type = (
            comp.get("status", {}).get("type", {}).get("name", "")
        )

        winner_abbr: str | None = None
        if status_type == "STATUS_FINAL":
            h_score = home.get("score") or 0
            a_score = away.get("score") or 0
            if h_score > a_score:
                winner_abbr = home.get("abbr")
            elif a_score > h_score:
                winner_abbr = away.get("abbr")

        game_date = date or datetime.now(timezone.utc).strftime("%Y%m%d")

        games.append({
            "home_abbr":   home.get("abbr", ""),
            "away_abbr":   away.get("abbr", ""),
            "home_name":   home.get("name", ""),
            "away_name":   away.get("name", ""),
            "status":      status_type,
            "home_score":  home.get("score"),
            "away_score":  away.get("score"),
            "winner_abbr": winner_abbr,
            "game_date":   game_date,
        })

    return games


def find_game(ticker: str, search_days: int = 2) -> dict | None:
    """
    Find an ESPN game matching a KXNBAGAME Kalshi ticker.

    Searches today + (search_days-1) prior days to handle games logged
    before/after midnight.  Returns the first match or None.

    Matching logic: both team abbreviations must appear (in either order)
    in the ESPN game's home_abbr / away_abbr fields.
    """
    parsed = _parse_ticker(ticker)
    if parsed is None:
        return None
    home_k, away_k = parsed
    espn_home = _to_espn_abbr(home_k)
    espn_away = _to_espn_abbr(away_k)
    needle = {espn_home, espn_away}

    for date_str in _date_strings(search_days):
        for game in get_nba_scoreboard(date_str):
            game_teams = {game["home_abbr"].upper(), game["away_abbr"].upper()}
            if needle == game_teams or needle.issubset(game_teams):
                return game

    return None
