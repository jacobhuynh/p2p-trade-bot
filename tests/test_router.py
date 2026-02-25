"""
tests/test_router.py

Tests the router — the entry point that classifies incoming Kalshi WebSocket
trade messages and dispatches to the correct handler.

Pure logic: no external APIs, no API keys required.

Run: pytest tests/test_router.py -v -s
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.router import classify_market, route


# ── classify_market() ─────────────────────────────────────────────────────────

def test_classify_kxnbagame():
    assert classify_market("KXNBAGAME-26FEB19BKNCLE-BKN") == "GAME_WINNER"

def test_classify_kxnbagame_prefix_only():
    assert classify_market("KXNBAGAME") == "GAME_WINNER"

def test_classify_kxnbawins():
    assert classify_market("KXNBAWINS-25-BOS") == "TOTALS"

def test_classify_kxnbasgprop():
    assert classify_market("KXNBASGPROP-25JAN15LBJ-PTS30") == "PLAYER_PROP"

def test_classify_non_nba():
    assert classify_market("KXNFLGAME-26FEB19KCCIN") == "NON_NBA"
    assert classify_market("KXNHLGAME-BOSCOL") == "NON_NBA"
    assert classify_market("KXMLBGAME-NYYLAD") == "NON_NBA"

def test_classify_unknown_nba():
    """NBA ticker that doesn't match any known prefix → UNKNOWN."""
    assert classify_market("KXNBAOTHER-SOMEMARKET") == "UNKNOWN"

def test_classify_empty_string():
    assert classify_market("") == "NON_NBA"


# ── route() ───────────────────────────────────────────────────────────────────

LONGSHOT_GAME_WINNER = {
    "market_ticker": "KXNBAGAME-26FEB19BKNCLE-BKN",
    "yes_price": 14,
}

MIDPRICE_GAME_WINNER = {
    "market_ticker": "KXNBAGAME-26FEB19LALGWS-LAL",
    "yes_price": 55,
}

TOTALS_TRADE = {
    "market_ticker": "KXNBAWINS-25-BOS",
    "yes_price": 40,
}

PLAYER_PROP_TRADE = {
    "market_ticker": "KXNBASGPROP-25JAN15LBJ-PTS30",
    "yes_price": 60,
}

NON_NBA_TRADE = {
    "market_ticker": "KXNFLGAME-26FEB19KCCIN",
    "yes_price": 12,
}

MOCK_TRADE_PACKET = {
    "ticker": "KXNBAGAME-26FEB19BKNCLE-BKN",
    "action": "BET_NO",
    "market_price": 14,
    "category": "NBA",
    "market_title": "Brooklyn at Cleveland Winner?",
    "market_type": "binary",
    "rules_primary": "Unknown",
    "no_sub_title": "Unknown",
}


def test_route_game_winner_calls_bouncer():
    """KXNBAGAME ticker → bouncer.process_trade is called, returns GAME_WINNER."""
    with patch("src.pipeline.bouncer.process_trade", return_value=MOCK_TRADE_PACKET) as mock_bouncer:
        market_type, packet = route(LONGSHOT_GAME_WINNER)

    assert market_type == "GAME_WINNER"
    assert packet == MOCK_TRADE_PACKET
    mock_bouncer.assert_called_once_with(LONGSHOT_GAME_WINNER)


def test_route_game_winner_midprice_returns_none():
    """Bouncer filters mid-price → route returns (GAME_WINNER, None)."""
    with patch("src.pipeline.bouncer.process_trade", return_value=None) as mock_bouncer:
        market_type, packet = route(MIDPRICE_GAME_WINNER)

    assert market_type == "GAME_WINNER"
    assert packet is None
    mock_bouncer.assert_called_once()


def test_route_totals_no_bouncer():
    """KXNBAWINS ticker → bouncer is never called, returns (TOTALS, None)."""
    with patch("src.pipeline.bouncer.process_trade") as mock_bouncer:
        market_type, packet = route(TOTALS_TRADE)

    assert market_type == "TOTALS"
    assert packet is None
    mock_bouncer.assert_not_called()


def test_route_player_prop_no_bouncer():
    """KXNBASGPROP ticker → bouncer is never called, returns (PLAYER_PROP, None)."""
    with patch("src.pipeline.bouncer.process_trade") as mock_bouncer:
        market_type, packet = route(PLAYER_PROP_TRADE)

    assert market_type == "PLAYER_PROP"
    assert packet is None
    mock_bouncer.assert_not_called()


def test_route_non_nba_silent_drop():
    """Non-NBA ticker → silently dropped, bouncer never called."""
    with patch("src.pipeline.bouncer.process_trade") as mock_bouncer:
        market_type, packet = route(NON_NBA_TRADE)

    assert market_type == "NON_NBA"
    assert packet is None
    mock_bouncer.assert_not_called()


def test_route_uses_market_ticker_field():
    """route() reads 'market_ticker' key from the trade dict (Kalshi format)."""
    trade = {"market_ticker": "KXNBAGAME-26FEB19BKNCLE-BKN", "yes_price": 14}
    with patch("src.pipeline.bouncer.process_trade", return_value=MOCK_TRADE_PACKET):
        market_type, _ = route(trade)
    assert market_type == "GAME_WINNER"


def test_route_uses_ticker_field():
    """route() falls back to 'ticker' key if 'market_ticker' is absent."""
    trade = {"ticker": "KXNBAGAME-26FEB19BKNCLE-BKN", "yes_price": 14}
    with patch("src.pipeline.bouncer.process_trade", return_value=MOCK_TRADE_PACKET):
        market_type, _ = route(trade)
    assert market_type == "GAME_WINNER"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_classify_kxnbagame()
    test_classify_kxnbawins()
    test_classify_kxnbasgprop()
    test_classify_non_nba()
    test_classify_unknown_nba()
    test_classify_empty_string()
    test_route_game_winner_calls_bouncer()
    test_route_game_winner_midprice_returns_none()
    test_route_totals_no_bouncer()
    test_route_player_prop_no_bouncer()
    test_route_non_nba_silent_drop()
    test_route_uses_market_ticker_field()
    test_route_uses_ticker_field()
    print("\nOK All router tests passed.")
