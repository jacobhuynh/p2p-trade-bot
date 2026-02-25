"""
tests/test_websocket.py

Integration tests for KalshiWebsocketClient.handle_message().

Tests the message handling loop WITHOUT making a real WebSocket connection.
Injects pre-formed message dicts directly into handle_message() to validate:
  - Routing decisions (which message types reach the pipeline)
  - APPROVED trades are logged; PASS/VETOED are not
  - Auth header structure is correct

Requirements:
  - KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set in .env
  - ANTHROPIC_API_KEY must be set for the full-pipeline test

Run: pytest tests/test_websocket.py -v -s
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# ── Skip guard — all tests in this file need Kalshi credentials ───────────────

_KALSHI_MISSING = not (
    os.getenv("KALSHI_API_KEY_ID") and
    os.getenv("KALSHI_PRIVATE_KEY_PATH") and
    Path(os.getenv("KALSHI_PRIVATE_KEY_PATH", "")).exists()
)

pytestmark = pytest.mark.skipif(
    _KALSHI_MISSING,
    reason="Kalshi credentials (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH) not set or key file missing"
)


# ── Client factory ────────────────────────────────────────────────────────────

def _make_client(tmp_db: str):
    """
    Instantiate a real KalshiWebsocketClient, then swap in a temp TradeLogger
    so no test trades land in data/live_trades.db.
    """
    from src.pipeline.websocket_client import KalshiWebsocketClient
    from src.execution.trade_logger import TradeLogger
    client = KalshiWebsocketClient()
    client.logger = TradeLogger(db_path=tmp_db)
    return client


# ── Sample messages ───────────────────────────────────────────────────────────

def _game_winner_msg(yes_price: int = 14, ticker: str = "KXNBAGAME-26FEB19BKNCLE-BKN") -> dict:
    return {
        "type": "trade",
        "msg": {"market_ticker": ticker, "yes_price": yes_price},
    }

TOTALS_MSG = {
    "type": "trade",
    "msg": {"market_ticker": "KXNBAWINS-25-BOS", "yes_price": 40},
}

NON_TRADE_MSG = {
    "type": "orderbook_delta",
    "msg": {"market_ticker": "KXNBAGAME-26FEB19BKNCLE-BKN", "yes_price": 14},
}

MIDPRICE_MSG = _game_winner_msg(yes_price=55)


# ── Auth header test ──────────────────────────────────────────────────────────

def test_auth_headers_structure():
    """
    _generate_auth_headers() should return a dict with the three required
    Kalshi authentication keys (no actual WS connection needed).
    """
    tmp = tempfile.mktemp(prefix="p2p_ws_test_", suffix=".db")
    try:
        client = _make_client(tmp)
        headers = client._generate_auth_headers()
        assert "KALSHI-ACCESS-KEY"       in headers, "Missing KALSHI-ACCESS-KEY"
        assert "KALSHI-ACCESS-SIGNATURE" in headers, "Missing KALSHI-ACCESS-SIGNATURE"
        assert "KALSHI-ACCESS-TIMESTAMP" in headers, "Missing KALSHI-ACCESS-TIMESTAMP"
        assert headers["KALSHI-ACCESS-KEY"] == os.getenv("KALSHI_API_KEY_ID")
        print(f"   Auth headers present: {list(headers.keys())}")
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── handle_message: routing / gating tests ───────────────────────────────────

@pytest.mark.asyncio
async def test_handle_non_trade_message_ignored():
    """
    Messages with type != 'trade' should be silently ignored.
    The analyst pipeline must never be called.
    """
    tmp = tempfile.mktemp(prefix="p2p_ws_test_", suffix=".db")
    try:
        client = _make_client(tmp)
        spy = MagicMock()
        client.analyst = spy

        await client.handle_message(NON_TRADE_MSG)

        spy.analyze_signal.assert_not_called()
        print("   Non-trade message correctly ignored")
    finally:
        Path(tmp).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_handle_totals_message_no_pipeline():
    """
    KXNBAWINS (Totals) ticker → placeholder handler → analyst NOT called.
    """
    tmp = tempfile.mktemp(prefix="p2p_ws_test_", suffix=".db")
    try:
        client = _make_client(tmp)
        spy = MagicMock()
        client.analyst = spy

        await client.handle_message(TOTALS_MSG)

        spy.analyze_signal.assert_not_called()
        print("   Totals message routed to placeholder without calling analyst")
    finally:
        Path(tmp).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_handle_mid_price_game_winner_no_pipeline():
    """
    KXNBAGAME with yes_price=55 → bouncer filters it (no longshot edge)
    → analyst is never called.

    Uses the real Kalshi REST call inside the bouncer.
    """
    tmp = tempfile.mktemp(prefix="p2p_ws_test_", suffix=".db")
    try:
        client = _make_client(tmp)
        spy = MagicMock()
        client.analyst = spy

        await client.handle_message(MIDPRICE_MSG)

        spy.analyze_signal.assert_not_called()
        print("   Mid-price GAME_WINNER filtered by bouncer, analyst not called")
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── handle_message: full pipeline test (real Claude) ─────────────────────────

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping real Claude pipeline test"
)
async def test_handle_game_winner_longshot_full_pipeline():
    """
    Inject a longshot GAME_WINNER message (yes_price=14) and run the full
    real pipeline: Kalshi REST → DuckDB → Claude Quant → Claude Orchestrator
    → possibly Claude Critic → temp TradeLogger.

    Asserts:
      - The pipeline returns a valid status (APPROVED / VETOED / PASS)
      - If APPROVED: temp DB has exactly 1 logged trade
      - If PASS/VETOED: temp DB has 0 trades (logger.log_trade not called)
    """
    tmp = tempfile.mktemp(prefix="p2p_ws_test_", suffix=".db")
    try:
        client = _make_client(tmp)

        msg = _game_winner_msg(yes_price=14, ticker="KXNBAGAME-26FEB19BKNCLE-BKN")
        await client.handle_message(msg)

        open_trades = client.logger.open_trades()
        n_logged = len(open_trades)

        # The pipeline must have produced one of the three valid outcomes
        assert n_logged in (0, 1), f"Expected 0 or 1 logged trades, got {n_logged}"

        if n_logged == 1:
            trade = open_trades[0]
            assert trade["status"] == "PENDING_RESOLUTION"
            assert trade["action"] in ("BET_YES", "BET_NO")
            assert trade["cost_usd"] > 0
            print(f"   APPROVED → trade #{trade['id']} logged: {trade['ticker']} {trade['action']} cost=${trade['cost_usd']:.2f}")
        else:
            print("   PASS or VETOED → no trade logged (correct)")

    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if _KALSHI_MISSING:
        print("SKIP: KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set. Skipping websocket tests.")
        sys.exit(0)

    print("\n=== Auth header ===")
    test_auth_headers_structure()

    print("\n=== Routing / gating (spy on analyst) ===")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_handle_non_trade_message_ignored())
    loop.run_until_complete(test_handle_totals_message_no_pipeline())
    loop.run_until_complete(test_handle_mid_price_game_winner_no_pipeline())

    if os.getenv("ANTHROPIC_API_KEY"):
        print("\n=== Full pipeline (real Claude) ===")
        loop.run_until_complete(test_handle_game_winner_longshot_full_pipeline())
    else:
        print("\nSKIP: ANTHROPIC_API_KEY not set — skipping full pipeline test")

    print("\nOK All websocket tests passed.")
