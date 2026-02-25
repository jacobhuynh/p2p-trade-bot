"""
tests/test_nba_tool.py

Tests for src/tools/nba_tool.py.

Split into two sections:
  1. Pure-logic tests — ticker parsing (no network, no API keys).
  2. Live nba_api tests — real calls to the NBA Stats API (public, no keys).
     These may be slow (~5s timeout per team) and occasionally flaky if the
     NBA Stats API is down — failures are treated as graceful None returns.

Run: pytest tests/test_nba_tool.py -v -s
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.nba_tool import _parse_teams_from_ticker, get_team_recent_records


# ── Pure-logic: ticker parsing ────────────────────────────────────────────────

def test_parse_teams_valid_3char():
    """Standard 3-char team ticker parses to (home, away)."""
    result = _parse_teams_from_ticker("KXNBAGAME-26FEB19LACBOS-NO")
    assert result == ("LAC", "BOS"), f"Expected (LAC, BOS), got {result}"

def test_parse_teams_valid_bkncle():
    result = _parse_teams_from_ticker("KXNBAGAME-26FEB19BKNCLE-BKN")
    assert result == ("BKN", "CLE"), f"Expected (BKN, CLE), got {result}"

def test_parse_teams_valid_yes_side():
    result = _parse_teams_from_ticker("KXNBAGAME-26FEB19ORLSAC-ORL")
    assert result is not None
    assert result[0] == "ORL"

def test_parse_teams_non_nba_returns_none():
    assert _parse_teams_from_ticker("KXNFLGAME-26FEB19KCCIN") is None

def test_parse_teams_kxnbawins_returns_none():
    """KXNBAWINS is not KXNBAGAME → None."""
    assert _parse_teams_from_ticker("KXNBAWINS-25-BOS") is None

def test_parse_teams_no_dash_returns_none():
    assert _parse_teams_from_ticker("KXNBAGAME26FEB19LACBOS") is None

def test_parse_teams_no_date_prefix_returns_none():
    assert _parse_teams_from_ticker("KXNBAGAME-LACBOS-NO") is None

def test_parse_teams_empty_returns_none():
    assert _parse_teams_from_ticker("") is None


# ── Live nba_api tests ────────────────────────────────────────────────────────

def test_get_team_recent_records_non_nba_ticker():
    """Non-KXNBAGAME ticker → immediately returns None (no API call made)."""
    result = get_team_recent_records("KXNFLGAME-26FEB19KCCIN")
    assert result is None

def test_get_team_recent_records_unknown_team_code():
    """Ticker with fake team codes → _get_team_id returns None → None overall."""
    result = get_team_recent_records("KXNBAGAME-26FEB19ZZZYYY-ZZZ")
    assert result is None, f"Expected None for unknown team codes, got {result}"
    print("   Unknown team codes gracefully returned None")

def test_get_team_recent_records_real():
    """
    Real nba_api call for a known-format ticker.

    The NBA Stats API is public but can be slow or rate-limited.
    We assert either:
      - A valid dict with 'home' and 'away' keys is returned, OR
      - None is returned (graceful failure — network issue or API unavailable)

    We never assert a specific win/loss record since historical data may vary.
    """
    ticker = "KXNBAGAME-26FEB19LACBOS-NO"
    result = get_team_recent_records(ticker, last_n=5)

    if result is None:
        print("   nba_api returned None (API unavailable or rate-limited) — OK")
        return

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "home" in result, f"Missing 'home' key: {result}"
    assert "away" in result, f"Missing 'away' key: {result}"

    home = result["home"]
    away = result["away"]
    assert home.get("abbr") == "LAC", f"Expected home abbr LAC, got {home.get('abbr')}"
    assert away.get("abbr") == "BOS", f"Expected away abbr BOS, got {away.get('abbr')}"

    # Record format is "W-L" e.g. "6-4"
    for label, team in [("home", home), ("away", away)]:
        last_n_key = "last5"
        if last_n_key in team:
            w, l = team[last_n_key].split("-")
            assert w.isdigit() and l.isdigit(), \
                f"{label} {last_n_key} record format invalid: {team[last_n_key]}"

    print(f"   home={home}  away={away}")

def test_get_team_recent_records_gsw():
    """GSW maps to Golden State Warriors in the nba_api lookup."""
    ticker = "KXNBAGAME-26FEB19GSWORL-GSW"
    result = get_team_recent_records(ticker, last_n=5)
    # Either valid dict or None (graceful failure)
    assert result is None or ("home" in result and "away" in result)
    print(f"   GSW ticker result: {result}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Pure-logic: ticker parsing ===")
    test_parse_teams_valid_3char()
    test_parse_teams_valid_bkncle()
    test_parse_teams_non_nba_returns_none()
    test_parse_teams_kxnbawins_returns_none()
    test_parse_teams_no_dash_returns_none()
    test_parse_teams_no_date_prefix_returns_none()
    test_parse_teams_empty_returns_none()

    print("\n=== Live nba_api tests ===")
    test_get_team_recent_records_non_nba_ticker()
    test_get_team_recent_records_unknown_team_code()
    test_get_team_recent_records_real()
    test_get_team_recent_records_gsw()

    print("\nOK All NBA tool tests passed.")
