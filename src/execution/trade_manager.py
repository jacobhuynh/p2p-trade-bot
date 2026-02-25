"""
src/execution/trade_manager.py

Paper trading simulator.  Records simulated trades and equity curve to local
files under data/paper/ (or PAPER_DATA_DIR).  Zero network calls.

Files written:
  book.json    — current portfolio state (cash + open positions + realized PnL)
  trades.csv   — append-only trade log
  equity.csv   — append-only equity-curve snapshot written after every trade

Accounting convention:
  - Cash decreases when we open a position (we pay the contract price).
  - Equity column in equity.csv = cash on hand.  This shrinks as capital is
    deployed and would grow on profitable closes (not yet implemented).
  - For BET_NO: we buy NO contracts at (100 - yes_price) cents each.
  - For BET_YES: we buy YES contracts at yes_price cents each.

Usage:
    from src.execution.trade_manager import PaperTradeManager
    report = PaperTradeManager().execute(decision, trade_packet)
"""

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import PAPER_STARTING_CASH, PAPER_MAX_CONTRACTS

# ── CSV column definitions ────────────────────────────────────────────────────
_TRADES_COLS = [
    "timestamp", "market_ticker", "side", "action",
    "price", "contracts", "notional", "cash_after", "note",
]
_EQUITY_COLS = [
    "timestamp", "cash", "unrealized_pnl", "realized_pnl", "equity", "n_positions",
]


class PaperTradeManager:
    """
    Simulates trade execution without touching any real exchange.

    Parameters
    ----------
    data_dir : str, optional
        Override the directory where book.json / trades.csv / equity.csv are
        stored.  Defaults to the PAPER_DATA_DIR env var, or "data/paper".
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or os.getenv("PAPER_DATA_DIR", "data/paper"))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.book_path   = self.data_dir / "book.json"
        self.trades_path = self.data_dir / "trades.csv"
        self.equity_path = self.data_dir / "equity.csv"

    # ── Persistence ───────────────────────────────────────────────────────────

    def load_book(self) -> dict:
        """Load portfolio state from disk, or return a fresh default."""
        if self.book_path.exists():
            with open(self.book_path) as f:
                return json.load(f)
        return {
            "cash":         PAPER_STARTING_CASH,
            "positions":    {},
            "realized_pnl": 0.0,
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        }

    def save_book(self, book: dict) -> None:
        """Persist portfolio state to disk."""
        book["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.book_path, "w") as f:
            json.dump(book, f, indent=2)

    # ── Mark-to-market ────────────────────────────────────────────────────────

    def mark_to_market(self, book: dict, price_map: dict) -> float:
        """
        Compute unrealized PnL given a current {ticker: yes_price_cents} map.

        For each open position:
          YES position: value = contracts * current_yes_price / 100
          NO  position: value = contracts * (100 - current_yes_price) / 100
          unrealized = position_value - cost_basis
        """
        unrealized = 0.0
        for ticker, pos in book.get("positions", {}).items():
            if ticker not in price_map:
                continue
            current_yes = price_map[ticker]
            current_price = (100 - current_yes) if pos["side"] == "NO" else current_yes
            position_value = pos["contracts"] * current_price / 100
            cost_basis     = pos["contracts"] * pos["avg_price"] / 100
            unrealized    += position_value - cost_basis
        return round(unrealized, 4)

    # ── Internal CSV helpers ──────────────────────────────────────────────────

    def _append_trade_row(self, row: dict) -> None:
        write_header = not self.trades_path.exists()
        with open(self.trades_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADES_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_equity_row(self, book: dict, unrealized_pnl: float = 0.0) -> None:
        """
        Snapshot equity state after every trade.
        equity = cash on hand (capital not yet deployed into positions).
        """
        write_header = not self.equity_path.exists()
        equity = book["cash"]   # cash balance; shrinks as positions are opened
        row = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "cash":           round(book["cash"], 4),
            "unrealized_pnl": round(unrealized_pnl, 4),
            "realized_pnl":   round(book["realized_pnl"], 4),
            "equity":         round(equity, 4),
            "n_positions":    len(book["positions"]),
        }
        with open(self.equity_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_EQUITY_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ── Core execution ────────────────────────────────────────────────────────

    def execute(self, decision: dict, trade_packet: dict) -> dict:
        """
        Simulate filling an APPROVED trade.

        Parameters
        ----------
        decision     : the final decision dict from LeadAnalyst.analyze_signal()
        trade_packet : the enriched trade packet from bouncer.process_trade()

        Returns
        -------
        execution_report dict with keys:
          status, ticker, side, contracts, entry_price_cents, notional,
          cash_before, cash_after, note
        """
        ticker    = trade_packet.get("ticker") or trade_packet.get("market_ticker")
        yes_price = trade_packet.get("market_price") or trade_packet.get("yes_price")
        action    = decision.get("action")        # "BET_YES" | "BET_NO"
        kelly     = decision.get("kelly_fraction") or 0.0

        if action not in ("BET_YES", "BET_NO"):
            return {"status": "SKIPPED", "reason": f"Action {action!r} is not a BET."}

        # ── Effective entry price per contract ───────────────────────────────
        # BET_NO  → buy NO  at (100 - yes_price) cents
        # BET_YES → buy YES at yes_price cents
        side               = "NO" if action == "BET_NO" else "YES"
        entry_price_cents  = (100 - yes_price) if side == "NO" else yes_price
        cost_per_contract  = entry_price_cents / 100.0   # dollars

        book       = self.load_book()
        cash_now   = book["cash"]

        # Guard: not enough for even one contract
        if cost_per_contract <= 0 or cash_now < cost_per_contract:
            return {
                "status": "SKIPPED",
                "reason": f"Insufficient cash ({cash_now:.2f}) for one contract at {entry_price_cents}c.",
            }

        # ── Contract sizing ──────────────────────────────────────────────────
        # contracts = floor(equity * risk_fraction / cost_per_contract)
        # risk_fraction = min(kelly, 0.02) so we never risk more than 2% even
        # when kelly says more.
        risk_fraction = min(0.02, kelly if kelly > 0 else 0.02)
        raw           = cash_now * risk_fraction / cost_per_contract
        contracts     = max(1, min(int(math.floor(raw)), PAPER_MAX_CONTRACTS))

        # Reduce if sizing exceeds available cash
        note = ""
        notional   = round(contracts * cost_per_contract, 4)
        cash_after = round(cash_now - notional, 4)

        if cash_after < 0:
            contracts  = max(1, int(math.floor(cash_now / cost_per_contract)))
            notional   = round(contracts * cost_per_contract, 4)
            cash_after = round(cash_now - notional, 4)
            note       = "scaled_down:insufficient_cash"

        # ── Update position book ─────────────────────────────────────────────
        book["cash"] = cash_after
        existing     = book["positions"].get(ticker)

        if existing and existing["side"] == side:
            # Average-in to an existing position on the same side
            total     = existing["contracts"] + contracts
            avg_price = round(
                (existing["avg_price"] * existing["contracts"] + entry_price_cents * contracts)
                / total, 4,
            )
            book["positions"][ticker] = {"side": side, "contracts": total, "avg_price": avg_price}
        else:
            book["positions"][ticker] = {
                "side":      side,
                "contracts": contracts,
                "avg_price": entry_price_cents,
            }

        self.save_book(book)

        # ── Append to logs ───────────────────────────────────────────────────
        ts = datetime.now(timezone.utc).isoformat()
        self._append_trade_row({
            "timestamp":     ts,
            "market_ticker": ticker,
            "side":          side,
            "action":        action,
            "price":         entry_price_cents,
            "contracts":     contracts,
            "notional":      notional,
            "cash_after":    cash_after,
            "note":          note,
        })
        self._append_equity_row(book)

        return {
            "status":            "FILLED",
            "ticker":            ticker,
            "side":              side,
            "contracts":         contracts,
            "entry_price_cents": entry_price_cents,
            "notional":          notional,
            "cash_before":       round(cash_now, 4),
            "cash_after":        cash_after,
            "note":              note,
        }


class LiveTradeManager:
    """
    Placeholder for real Kalshi order execution.
    Raises NotImplementedError immediately so nothing is accidentally traded.
    """

    def execute(self, decision: dict, trade_packet: dict) -> dict:
        raise NotImplementedError(
            "Live execution is not yet implemented. "
            "Set EXECUTION_MODE=paper (or unset it) to use the paper simulator."
        )


