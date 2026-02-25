"""
src/execution/trade_logger.py

SQLite-backed trade logger for the live mock trader.

When the pipeline APPROVES a trade during a live WebSocket session, call
log_trade() to persist the full decision to data/live_trades.db with
status=PENDING_RESOLUTION.

Later, run `python -m src.settle` to query the ESPN scoreboard API for
resolved games and evaluate each PENDING_RESOLUTION trade — computing
the hypothetical P&L and updating status to EVALUATED.
"""

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_DB = "data/live_trades.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS live_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    market_title    TEXT,
    action          TEXT    NOT NULL,   -- BET_YES | BET_NO
    side            TEXT    NOT NULL,   -- yes | no
    yes_price       INTEGER NOT NULL,
    entry_cents     INTEGER NOT NULL,   -- cost per contract (cents)
    contracts       INTEGER NOT NULL,
    cost_usd        REAL    NOT NULL,
    kelly           REAL,
    confidence      TEXT,
    calibration_gap REAL,
    sample_size     INTEGER,
    verdict         TEXT,
    risk_score      INTEGER,
    concerns        TEXT,               -- JSON array
    status          TEXT    NOT NULL DEFAULT 'PENDING_RESOLUTION',  -- PENDING_RESOLUTION | EVALUATED
    result          TEXT,              -- yes | no once market resolves
    payout_usd      REAL,
    pnl_usd         REAL,
    evaluated_at    TEXT
)
"""


class TradeLogger:
    def __init__(self, db_path: str = _DEFAULT_DB):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with self._conn() as con:
            con.execute(_CREATE_TABLE)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ── Write ────────────────────────────────────────────────────────────────

    def log_trade(self, decision: dict, trade_packet: dict, stake: float = 10.0) -> int:
        """
        Insert a new PENDING_RESOLUTION trade row.  Returns the auto-incremented row id.

        Position sizing:
          entry_cents = yes_price        for BET_YES
          entry_cents = 100 - yes_price  for BET_NO
          contracts   = max(1, floor(stake / (entry_cents / 100)))
          cost_usd    = contracts * entry_cents / 100
        """
        action      = decision.get("action", "")
        side        = decision.get("side", "")
        yes_price   = int(decision.get("price", 0))
        entry_cents = yes_price if action == "BET_YES" else (100 - yes_price)
        contracts   = max(1, math.floor(stake / (entry_cents / 100)))
        cost_usd    = round(contracts * entry_cents / 100, 4)

        quant       = decision.get("quant_summary", {})
        critic      = decision.get("critic", {})
        concerns    = json.dumps(critic.get("concerns") or [])

        with self._conn() as con:
            cur = con.execute(
                """
                INSERT INTO live_trades (
                    logged_at, ticker, market_title, action, side,
                    yes_price, entry_cents, contracts, cost_usd,
                    kelly, confidence, calibration_gap, sample_size, verdict,
                    risk_score, concerns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    decision.get("ticker") or trade_packet.get("ticker"),
                    trade_packet.get("market_title"),
                    action,
                    side,
                    yes_price,
                    entry_cents,
                    contracts,
                    cost_usd,
                    decision.get("kelly_fraction"),
                    decision.get("confidence"),
                    quant.get("calibration_gap"),
                    quant.get("sample_size"),
                    quant.get("verdict"),
                    critic.get("risk_score"),
                    concerns,
                ),
            )
            return cur.lastrowid

    def evaluate_trade(self, trade_id: int, result: str) -> dict:
        """
        Mark a trade as EVALUATED using the known game result ('yes' or 'no').

        Returns a dict with trade_id, won, payout_usd, and pnl_usd.
        """
        with self._conn() as con:
            row = con.execute(
                "SELECT side, contracts, cost_usd FROM live_trades WHERE id = ?",
                (trade_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No trade with id={trade_id}")

            side, contracts, cost_usd = row["side"], row["contracts"], row["cost_usd"]
            won    = (side == "yes" and result == "yes") or (side == "no" and result == "no")
            payout = round(contracts * 1.0, 4) if won else 0.0
            pnl    = round(payout - cost_usd, 4)

            con.execute(
                """
                UPDATE live_trades
                SET status       = 'EVALUATED',
                    result       = ?,
                    payout_usd   = ?,
                    pnl_usd      = ?,
                    evaluated_at = ?
                WHERE id = ?
                """,
                (result, payout, pnl, datetime.now(timezone.utc).isoformat(), trade_id),
            )
            return {"trade_id": trade_id, "won": won, "payout_usd": payout, "pnl_usd": pnl}

    # ── Read ─────────────────────────────────────────────────────────────────

    def open_trades(self) -> list[dict]:
        """Return all PENDING_RESOLUTION rows as plain dicts."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM live_trades WHERE status = 'PENDING_RESOLUTION' ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def summary(self) -> dict:
        """P&L summary across all EVALUATED trades."""
        with self._conn() as con:
            row = con.execute(
                """
                SELECT
                    COUNT(*)                                    AS n_trades,
                    SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) AS n_wins,
                    COALESCE(SUM(pnl_usd), 0.0)                AS total_pnl,
                    COALESCE(SUM(cost_usd), 0.0)               AS total_staked
                FROM live_trades
                WHERE status = 'EVALUATED'
                """
            ).fetchone()
            n_trades     = row["n_trades"]
            n_wins       = row["n_wins"] or 0
            total_pnl    = row["total_pnl"]
            total_staked = row["total_staked"]
            return {
                "n_trades":    n_trades,
                "n_wins":      n_wins,
                "total_pnl":   round(total_pnl, 4),
                "win_rate":    round(n_wins / n_trades, 4) if n_trades else 0.0,
                "roi":         round(total_pnl / total_staked, 4) if total_staked else 0.0,
            }
