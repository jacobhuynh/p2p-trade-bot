"""
tests/test_settle.py

Tests for src/settle.py — the P&L resolution module.

Split into two sections:
  1. Pure-logic tests for _determine_result() — no network, no API keys.
  2. Live run_settle() tests — real ESPN API calls (public, no keys required).
     Uses a temp SQLite DB to avoid touching data/live_trades.db.

Run: pytest tests/test_settle.py -v -s
"""

import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.settle import _determine_result, run_settle
from src.execution.trade_logger import TradeLogger


# ── Pure-logic: _determine_result() ──────────────────────────────────────────

def test_determine_result_home_wins():
    """Home team wins → YES resolves 'yes' (Kalshi convention for KXNBAGAME)."""
    game = {"winner_abbr": "LAC", "home_abbr": "LAC", "away_abbr": "BOS"}
    trade = {}  # _determine_result doesn't use trade fields currently
    assert _determine_result(trade, game) == "yes"

def test_determine_result_away_wins():
    """Away team wins → YES resolves 'no'."""
    game = {"winner_abbr": "BOS", "home_abbr": "LAC", "away_abbr": "BOS"}
    trade = {}
    assert _determine_result(trade, game) == "no"

def test_determine_result_no_winner():
    """winner_abbr is empty → cannot determine result → None."""
    game = {"winner_abbr": "", "home_abbr": "LAC", "away_abbr": "BOS"}
    trade = {}
    assert _determine_result(trade, game) is None

def test_determine_result_winner_abbr_missing():
    """winner_abbr key absent → None."""
    game = {"home_abbr": "LAC", "away_abbr": "BOS"}
    trade = {}
    assert _determine_result(trade, game) is None

def test_determine_result_unknown_team():
    """Winner doesn't match either team in the game dict → None."""
    game = {"winner_abbr": "GSW", "home_abbr": "LAC", "away_abbr": "BOS"}
    trade = {}
    assert _determine_result(trade, game) is None

def test_determine_result_case_insensitive():
    """Comparison is uppercased on both sides."""
    game = {"winner_abbr": "lac", "home_abbr": "LAC", "away_abbr": "BOS"}
    trade = {}
    assert _determine_result(trade, game) == "yes"


# ── Helpers for planting trades in a temp DB ──────────────────────────────────

def _plant_pending_trade(db_path: str, ticker: str, action: str = "BET_NO",
                          side: str = "no", yes_price: int = 14) -> int:
    """Insert a PENDING_RESOLUTION row directly into a temp DB and return its id."""
    entry_cents = yes_price if action == "BET_YES" else (100 - yes_price)
    contracts = 10
    cost_usd = round(contracts * entry_cents / 100, 4)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at       TEXT    NOT NULL,
            ticker          TEXT    NOT NULL,
            market_title    TEXT,
            action          TEXT    NOT NULL,
            side            TEXT    NOT NULL,
            yes_price       INTEGER NOT NULL,
            entry_cents     INTEGER NOT NULL,
            contracts       INTEGER NOT NULL,
            cost_usd        REAL    NOT NULL,
            kelly           REAL,
            confidence      TEXT,
            calibration_gap REAL,
            sample_size     INTEGER,
            verdict         TEXT,
            risk_score      INTEGER,
            concerns        TEXT,
            status          TEXT    NOT NULL DEFAULT 'PENDING_RESOLUTION',
            result          TEXT,
            payout_usd      REAL,
            pnl_usd         REAL,
            evaluated_at    TEXT
        )
    """)
    cur = con.execute(
        """INSERT INTO live_trades
           (logged_at, ticker, market_title, action, side, yes_price,
            entry_cents, contracts, cost_usd, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING_RESOLUTION')""",
        (datetime.now(timezone.utc).isoformat(), ticker, "Test Market",
         action, side, yes_price, entry_cents, contracts, cost_usd)
    )
    con.commit()
    trade_id = cur.lastrowid
    con.close()
    return trade_id


# ── Live run_settle() tests ───────────────────────────────────────────────────

def test_run_settle_no_pending_trades():
    """Empty DB → prints 'No pending trades', returns without error."""
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        # Create the DB/table by instantiating TradeLogger, then settle it
        TradeLogger(db_path=tmp)  # creates table
        run_settle(db_path=tmp)   # should print "No pending trades to evaluate."
        # If we get here without an exception, the test passes
    finally:
        Path(tmp).unlink(missing_ok=True)

    print("   Empty DB settled without error")


def test_run_settle_fake_ticker_stays_pending():
    """
    Plant a trade with a deliberately impossible ticker.
    ESPN find_game() should return None (no match) → trade stays PENDING.

    Uses the real ESPN API (public, no keys needed).
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        fake_ticker = "KXNBAGAME-99DEC99ZZZZZZ-ZZZ"
        trade_id = _plant_pending_trade(tmp, fake_ticker)

        run_settle(db_path=tmp)

        logger = TradeLogger(db_path=tmp)
        open_trades = logger.open_trades()
        assert len(open_trades) == 1, \
            f"Trade should still be PENDING — got {len(open_trades)} open trades"
        assert open_trades[0]["ticker"] == fake_ticker

        print(f"   Trade #{trade_id} ({fake_ticker}) correctly stayed PENDING")
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_run_settle_multiple_pending_all_fake():
    """
    Multiple pending trades, all with fake tickers → all stay PENDING.
    Verifies run_settle() loops over multiple trades without crashing.
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        tickers = [
            "KXNBAGAME-99DEC99AAABBB-AAA",
            "KXNBAGAME-99DEC99CCCDDD-CCC",
        ]
        for ticker in tickers:
            _plant_pending_trade(tmp, ticker)

        run_settle(db_path=tmp)

        logger = TradeLogger(db_path=tmp)
        open_trades = logger.open_trades()
        assert len(open_trades) == 2, \
            f"Both trades should remain PENDING — got {len(open_trades)}"

        print(f"   {len(open_trades)} trades correctly stayed PENDING")
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Pure-logic: _determine_result() ===")
    test_determine_result_home_wins()
    test_determine_result_away_wins()
    test_determine_result_no_winner()
    test_determine_result_winner_abbr_missing()
    test_determine_result_unknown_team()
    test_determine_result_case_insensitive()

    print("\n=== Live run_settle() tests ===")
    test_run_settle_no_pending_trades()
    test_run_settle_fake_ticker_stays_pending()
    test_run_settle_multiple_pending_all_fake()

    print("\nOK All settle tests passed.")
