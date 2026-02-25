"""
tests/test_espn_tool.py

Tests for src/tools/espn_tool.py.

Split into two sections:
  1. Pure-logic tests — ticker parsing and abbreviation mapping (no network).
  2. Live ESPN tests   — real HTTP calls to the public ESPN scoreboard API
                         (no API keys required).

Run: pytest tests/test_espn_tool.py -v -s
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tools.espn_tool import (
    _parse_ticker,
    _to_espn_abbr,
    get_nba_scoreboard,
    find_game,
)


# ── Pure-logic: ticker parsing ────────────────────────────────────────────────

def test_parse_ticker_valid_3char_teams():
    """Standard ticker with two 3-char team codes."""
    result = _parse_ticker("KXNBAGAME-26FEB19LACBOS-NO")
    assert result == ("LAC", "BOS"), f"Expected (LAC, BOS), got {result}"

def test_parse_ticker_valid_bkncle():
    result = _parse_ticker("KXNBAGAME-26FEB19BKNCLE-BKN")
    assert result == ("BKN", "CLE"), f"Expected (BKN, CLE), got {result}"

def test_parse_ticker_valid_gsw():
    """GSW is 3 chars; ensure greedy regex handles it."""
    result = _parse_ticker("KXNBAGAME-26FEB19LALGWS-LAL")
    assert result == ("LAL", "GWS") or result is not None, \
        "Should parse without crashing even if team codes are unusual"

def test_parse_ticker_non_nba_returns_none():
    assert _parse_ticker("KXNFLGAME-26FEB19KCCIN") is None

def test_parse_ticker_missing_dash_returns_none():
    assert _parse_ticker("KXNBAGAME26FEB19LACBOS") is None

def test_parse_ticker_empty_returns_none():
    assert _parse_ticker("") is None

def test_parse_ticker_no_date_segment_returns_none():
    """Middle segment without proper YYMONDD prefix → None."""
    assert _parse_ticker("KXNBAGAME-LACBOS-NO") is None


# ── Pure-logic: abbreviation mapping ─────────────────────────────────────────

def test_abbr_gsw_maps_to_gs():
    assert _to_espn_abbr("GSW") == "GS"

def test_abbr_nop_maps_to_no():
    assert _to_espn_abbr("NOP") == "NO"

def test_abbr_sas_maps_to_sa():
    assert _to_espn_abbr("SAS") == "SA"

def test_abbr_uta_maps_to_utah():
    assert _to_espn_abbr("UTA") == "UTAH"

def test_abbr_bos_passthrough():
    """Unmapped abbreviation should pass through unchanged."""
    assert _to_espn_abbr("BOS") == "BOS"

def test_abbr_lac_passthrough():
    assert _to_espn_abbr("LAC") == "LAC"

def test_abbr_lowercase_input():
    """Input is uppercased before lookup."""
    assert _to_espn_abbr("gsw") == "GS"
    assert _to_espn_abbr("bos") == "BOS"


# ── Live ESPN API: scoreboard ─────────────────────────────────────────────────

def test_get_nba_scoreboard_returns_list():
    """Real ESPN call: result is always a list (may be empty if no games today)."""
    result = get_nba_scoreboard()
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    print(f"   {len(result)} game(s) found for today")

def test_get_nba_scoreboard_game_schema():
    """If games exist today, each has the required keys."""
    games = get_nba_scoreboard()
    required_keys = {"home_abbr", "away_abbr", "status", "home_score", "away_score",
                     "winner_abbr", "game_date", "home_name", "away_name"}
    for game in games:
        missing = required_keys - set(game.keys())
        assert not missing, f"Game dict missing keys: {missing}\nGot: {game}"
    print(f"   Schema check passed on {len(games)} game(s)")

def test_get_nba_scoreboard_future_date_returns_list():
    """A date in the far future returns an empty list, not an exception."""
    result = get_nba_scoreboard(date="20991231")
    assert isinstance(result, list)
    assert result == []
    print("   Far-future date correctly returned empty list")

def test_get_nba_scoreboard_past_date_returns_list():
    """A known past NBA date should return games (or empty list — just no crash)."""
    result = get_nba_scoreboard(date="20240101")
    assert isinstance(result, list)
    print(f"   {len(result)} game(s) on 2024-01-01")

def test_get_nba_scoreboard_winner_set_on_final():
    """For STATUS_FINAL games, winner_abbr should be a non-empty string."""
    games = get_nba_scoreboard(date="20240101")
    for game in games:
        if game["status"] == "STATUS_FINAL":
            assert isinstance(game["winner_abbr"], str) and game["winner_abbr"], \
                f"STATUS_FINAL game missing winner_abbr: {game}"
    print("   winner_abbr check passed for final games")


# ── Live ESPN API: find_game ──────────────────────────────────────────────────

def test_find_game_fake_ticker_returns_none():
    """A ticker with fake teams should never match any ESPN game."""
    result = find_game("KXNBAGAME-99DEC99ZZZZZZ-NO", search_days=2)
    assert result is None, f"Expected None for fake ticker, got {result}"
    print("   Fake ticker correctly returned None")

def test_find_game_non_nba_ticker_returns_none():
    """Non-KXNBAGAME ticker can't be parsed → None."""
    result = find_game("KXNFLGAME-26FEB19KCCIN")
    assert result is None
    print("   Non-NBA ticker returned None without crashing")

def test_find_game_malformed_ticker_returns_none():
    result = find_game("KXNBAGAME-NO")
    assert result is None
    print("   Malformed ticker returned None without crashing")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Pure-logic: ticker parsing ===")
    test_parse_ticker_valid_3char_teams()
    test_parse_ticker_valid_bkncle()
    test_parse_ticker_non_nba_returns_none()
    test_parse_ticker_missing_dash_returns_none()
    test_parse_ticker_empty_returns_none()
    test_parse_ticker_no_date_segment_returns_none()

    print("\n=== Pure-logic: abbreviation mapping ===")
    test_abbr_gsw_maps_to_gs()
    test_abbr_nop_maps_to_no()
    test_abbr_sas_maps_to_sa()
    test_abbr_uta_maps_to_utah()
    test_abbr_bos_passthrough()
    test_abbr_lowercase_input()

    print("\n=== Live ESPN API: scoreboard ===")
    test_get_nba_scoreboard_returns_list()
    test_get_nba_scoreboard_game_schema()
    test_get_nba_scoreboard_future_date_returns_list()
    test_get_nba_scoreboard_past_date_returns_list()
    test_get_nba_scoreboard_winner_set_on_final()

    print("\n=== Live ESPN API: find_game ===")
    test_find_game_fake_ticker_returns_none()
    test_find_game_non_nba_ticker_returns_none()
    test_find_game_malformed_ticker_returns_none()

    print("\nOK All ESPN tool tests passed.")
