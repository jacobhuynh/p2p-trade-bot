"""
tests/test_report_trades.py

Tests for the reporting script that summarizes evaluated trades.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.execution.trade_logger import TradeLogger
from src.report_trades import run_report


def _init_db_with_schema(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.executescript(
        """
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
        );
        """
    )
    con.commit()
    con.close()


def _insert_trade(
    db_path: str,
    *,
    status: str,
    pnl_usd: float | None = None,
    result: str | None = None,
    ticker: str = "TCKR",
) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO live_trades (
            logged_at, ticker, market_title, action, side,
            yes_price, entry_cents, contracts, cost_usd,
            status, result, payout_usd, pnl_usd, evaluated_at
        ) VALUES (
            '2025-01-01T00:00:00Z', ?, 'Test Market', 'BET_NO', 'no',
            10, 90, 1, 0.9,
            ?, ?, ?, ?, '2025-01-02T00:00:00Z'
        )
        """,
        (ticker, status, result, (pnl_usd or 0.0), pnl_usd),
    )
    con.commit()
    con.close()


def test_run_report_no_evaluated_trades(capsys):
    tmp = tempfile.mktemp(prefix="p2p_report_test_", suffix=".db")
    try:
        _init_db_with_schema(tmp)
        # One pending trade, no evaluated
        _insert_trade(tmp, status="PENDING_RESOLUTION")

        run_report(db_path=tmp)
        out, _ = capsys.readouterr()
        assert "No evaluated trades" in out
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_run_report_with_evaluated_trades(capsys):
    tmp = tempfile.mktemp(prefix="p2p_report_test_", suffix=".db")
    try:
        _init_db_with_schema(tmp)
        # One win, one loss, plus one pending that should not be counted
        _insert_trade(tmp, status="EVALUATED", pnl_usd=1.0, result="no", ticker="GAME_A")
        _insert_trade(tmp, status="EVALUATED", pnl_usd=-0.5, result="yes", ticker="GAME_B")
        _insert_trade(tmp, status="PENDING_RESOLUTION")

        logger = TradeLogger(db_path=tmp)
        evaluated = logger.evaluated_trades()
        assert len(evaluated) == 2

        run_report(db_path=tmp)
        out, _ = capsys.readouterr()

        # Should mention evaluated trades and summary line components
        assert "Evaluated trades" in out
        summary = logger.summary()
        assert f"Trades     : {summary['n_trades']}" in out
        assert f"Total P&L  : ${summary['total_pnl']:+.2f}" in out
        # New bankroll statistics
        assert "Starting cash" in out
        assert "Total staked" in out
        # Market statistics
        assert "Markets:" in out
        assert "Unique markets" in out
    finally:
        Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    with patch("builtins.print") as mock_print:
        tmp_db = tempfile.mktemp(prefix="p2p_report_test_", suffix=".db")
        try:
            _init_db_with_schema(tmp_db)
            _insert_trade(tmp_db, status="EVALUATED", pnl_usd=2.0, result="no")
            run_report(db_path=tmp_db)
        finally:
            Path(tmp_db).unlink(missing_ok=True)

