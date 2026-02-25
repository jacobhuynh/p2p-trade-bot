"""
tests/test_settle.py

Tests for src/settle.py — the P&L resolution module.

All tests mock `kalshi_rest.get_market_details` so no real API calls are made
and no Kalshi credentials are required.

Run: pytest tests/test_settle.py -v -s
"""

import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.settle import run_settle
from src.execution.trade_logger import TradeLogger

PATCH_TARGET = "src.settle.get_market_details"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plant_pending_trade(db_path: str, ticker: str, action: str = "BET_NO",
                          side: str = "no", yes_price: int = 14) -> int:
    """Insert a PENDING_RESOLUTION row directly and return its id."""
    entry_cents = yes_price if action == "BET_YES" else (100 - yes_price)
    contracts   = 10
    cost_usd    = round(contracts * entry_cents / 100, 4)

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


# ── No pending trades ─────────────────────────────────────────────────────────

def test_run_settle_no_pending_trades():
    """Empty DB → prints 'No pending trades', Kalshi API never called."""
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        TradeLogger(db_path=tmp)
        with patch(PATCH_TARGET) as mock_api:
            run_settle(db_path=tmp)
            mock_api.assert_not_called()
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   Empty DB settled without error, no API call made")


# ── API unavailable ───────────────────────────────────────────────────────────

def test_run_settle_api_unavailable():
    """
    Kalshi credentials missing or network error → get_market_details returns None
    → trade stays PENDING.
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        _plant_pending_trade(tmp, "KXNBAGAME-26FEB19BKNCLE-BKN")
        with patch(PATCH_TARGET, return_value=None):
            run_settle(db_path=tmp)
        open_trades = TradeLogger(db_path=tmp).open_trades()
        assert len(open_trades) == 1
        assert open_trades[0]["status"] == "PENDING_RESOLUTION"
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   API unavailable → trade correctly stayed PENDING")


# ── Market not yet finalized ──────────────────────────────────────────────────

def test_run_settle_market_still_open():
    """Market status='open' → game still in progress → trade stays PENDING."""
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        _plant_pending_trade(tmp, "KXNBAGAME-26FEB19BKNCLE-BKN")
        with patch(PATCH_TARGET, return_value={"status": "open", "result": ""}):
            run_settle(db_path=tmp)
        open_trades = TradeLogger(db_path=tmp).open_trades()
        assert len(open_trades) == 1
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   status=open → trade correctly stayed PENDING")


def test_run_settle_market_closed_not_finalized():
    """Market status='closed' (trading ended, not yet settled) → stays PENDING."""
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        _plant_pending_trade(tmp, "KXNBAGAME-26FEB19BKNCLE-BKN")
        with patch(PATCH_TARGET, return_value={"status": "closed", "result": ""}):
            run_settle(db_path=tmp)
        open_trades = TradeLogger(db_path=tmp).open_trades()
        assert len(open_trades) == 1
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   status=closed → trade correctly stayed PENDING")


# ── Market finalized: win ─────────────────────────────────────────────────────

def test_run_settle_market_finalized_win():
    """
    Market finalized with result='no', trade side='no' (BET_NO) → WIN → EVALUATED
    with positive P&L.
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        ticker   = "KXNBAGAME-26FEB19BKNCLE-BKN"
        trade_id = _plant_pending_trade(tmp, ticker, action="BET_NO",
                                         side="no", yes_price=14)
        with patch(PATCH_TARGET, return_value={"status": "finalized", "result": "no"}):
            run_settle(db_path=tmp)

        logger = TradeLogger(db_path=tmp)
        assert len(logger.open_trades()) == 0, "Trade should no longer be PENDING"

        s = logger.summary()
        assert s["n_trades"] == 1
        assert s["n_wins"]   == 1
        assert s["total_pnl"] > 0
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   BET_NO + result=no → WIN, pnl > 0")


# ── Market finalized: loss ────────────────────────────────────────────────────

def test_run_settle_market_finalized_loss():
    """
    Market finalized with result='yes', trade side='no' (BET_NO) → LOSS → EVALUATED
    with negative P&L.
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        ticker = "KXNBAGAME-26FEB19BKNCLE-BKN"
        _plant_pending_trade(tmp, ticker, action="BET_NO", side="no", yes_price=14)
        with patch(PATCH_TARGET, return_value={"status": "finalized", "result": "yes"}):
            run_settle(db_path=tmp)

        logger = TradeLogger(db_path=tmp)
        assert len(logger.open_trades()) == 0

        s = logger.summary()
        assert s["n_trades"] == 1
        assert s["n_wins"]   == 0
        assert s["total_pnl"] < 0
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   BET_NO + result=yes → LOSS, pnl < 0")


# ── Multiple trades: mixed outcomes ──────────────────────────────────────────

def test_run_settle_multiple_mixed():
    """
    Two pending trades: one finalized (WIN), one still open.
    After settle: 1 EVALUATED, 1 still PENDING.
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        ticker_done = "KXNBAGAME-26FEB19BKNCLE-BKN"
        ticker_open = "KXNBAGAME-26FEB19LALGWS-LAL"
        _plant_pending_trade(tmp, ticker_done, action="BET_NO", side="no", yes_price=14)
        _plant_pending_trade(tmp, ticker_open, action="BET_NO", side="no", yes_price=18)

        def _mock_api(ticker):
            if ticker == ticker_done:
                return {"status": "finalized", "result": "no"}
            return {"status": "open", "result": ""}

        with patch(PATCH_TARGET, side_effect=_mock_api):
            run_settle(db_path=tmp)

        logger = TradeLogger(db_path=tmp)
        assert len(logger.open_trades()) == 1, "One trade should remain PENDING"
        assert logger.open_trades()[0]["ticker"] == ticker_open

        s = logger.summary()
        assert s["n_trades"] == 1   # only one EVALUATED
        assert s["n_wins"]   == 1
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   Mixed: 1 finalized WIN + 1 open → correct split")


# ── Unknown result string ─────────────────────────────────────────────────────

def test_run_settle_unrecognised_result():
    """
    Kalshi returns finalized but result is an unexpected string → stays PENDING.
    Guards against API changes.
    """
    tmp = tempfile.mktemp(prefix="p2p_settle_test_", suffix=".db")
    try:
        _plant_pending_trade(tmp, "KXNBAGAME-26FEB19BKNCLE-BKN")
        with patch(PATCH_TARGET, return_value={"status": "finalized", "result": "void"}):
            run_settle(db_path=tmp)
        open_trades = TradeLogger(db_path=tmp).open_trades()
        assert len(open_trades) == 1
    finally:
        Path(tmp).unlink(missing_ok=True)
    print("   Unrecognised result string → trade correctly stayed PENDING")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_run_settle_no_pending_trades()
    test_run_settle_api_unavailable()
    test_run_settle_market_still_open()
    test_run_settle_market_closed_not_finalized()
    test_run_settle_market_finalized_win()
    test_run_settle_market_finalized_loss()
    test_run_settle_multiple_mixed()
    test_run_settle_unrecognised_result()
    print("\nOK All settle tests passed.")
