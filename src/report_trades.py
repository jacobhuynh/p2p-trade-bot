"""
src/report_trades.py

Reporting script for the live mock trader.

Reads data/live_trades.db (or a user-specified SQLite file), finds all
EVALUATED trades, prints them in a human-readable table, and then prints
summary statistics using TradeLogger.summary().

Usage:
    python -m src.report_trades
    python -m src.report_trades --db data/live_trades.db
"""

import argparse
from collections import Counter
from typing import Iterable

from src.config import PAPER_STARTING_CASH
from src.execution.trade_logger import TradeLogger


def _format_trade_row(trade: dict) -> str:
    """Return a single-line string summary for an evaluated trade."""
    pnl = trade.get("pnl_usd")
    pnl_sign = "+" if pnl is not None and pnl >= 0 else ""
    return (
        f"[#{trade.get('id', ''):>3}] "
        f"{(trade.get('logged_at') or '')[:19]}  "
        f"{(trade.get('ticker') or ''):<32}  "
        f"{(trade.get('action') or ''):<7}  "
        f"{(trade.get('side') or ''):<3}  "
        f"{trade.get('yes_price', ''):>3}c  "
        f"contracts={trade.get('contracts', ''):<3}  "
        f"cost=${trade.get('cost_usd', 0.0):>6.2f}  "
        f"result={trade.get('result') or '':<3}  "
        f"pnl={pnl_sign}${(pnl or 0.0):.2f}"
    )


def _print_trades(trades: Iterable[dict]) -> None:
    print(f"\n{'='*94}")
    print(f"  Evaluated trades")
    print(f"{'='*94}")
    for trade in trades:
        print("  " + _format_trade_row(trade))
    print(f"{'-'*94}")


def run_report(db_path: str = "data/live_trades.db") -> None:
    """Entry point for generating a report on all evaluated trades."""
    logger = TradeLogger(db_path=db_path)
    evaluated = logger.evaluated_trades()

    if not evaluated:
        print(f"\nNo evaluated trades found in {db_path}.")
        return

    _print_trades(evaluated)

    summary = logger.summary()
    print("\nCumulative P&L on evaluated trades:")
    n_trades = summary["n_trades"]
    total_pnl = summary["total_pnl"]
    total_staked = summary.get("total_staked", 0.0)
    current_bankroll = PAPER_STARTING_CASH + total_pnl

    print(f"  Trades     : {n_trades}")
    print(
        f"  Win rate   : {summary['win_rate']*100:.1f}%  "
        f"({summary['n_wins']}/{summary['n_trades']})"
    )
    print(f"  Total P&L  : ${total_pnl:+.2f}")
    print(f"  ROI        : {summary['roi']*100:+.1f}%")

    # Bankroll and staking stats
    print(f"\nBankroll:")
    print(f"  Starting cash : ${PAPER_STARTING_CASH:,.2f}")
    print(f"  Total staked  : ${total_staked:,.2f}")
    print(f"  Current value : ${current_bankroll:,.2f}")

    if n_trades:
        avg_stake = total_staked / n_trades if total_staked else 0.0
        avg_pnl = total_pnl / n_trades
        pnls = [float(t.get("pnl_usd") or 0.0) for t in evaluated]
        max_win = max((p for p in pnls if p > 0), default=0.0)
        max_loss = min((p for p in pnls if p < 0), default=0.0)

        print(f"\nPer-trade stats:")
        print(f"  Avg stake    : ${avg_stake:,.2f}")
        print(f"  Avg P&L      : ${avg_pnl:,.2f}")
        print(f"  Max win      : ${max_win:,.2f}")
        print(f"  Max loss     : ${max_loss:,.2f}")

        # Markets / games we had stakes in
        tickers = [t.get("ticker") for t in evaluated if t.get("ticker")]
        counter = Counter(tickers)
        n_markets = len(counter)
        print(f"\nMarkets:")
        print(f"  Unique markets : {n_markets}")
        if n_markets:
            print(f"  Top markets    :")
            for ticker, count in counter.most_common(10):
                print(f"    {ticker}: {count} trade(s)")

    print(f"{'='*94}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report evaluated trades and P&L from live_trades.db"
    )
    parser.add_argument(
        "--db",
        dest="db_path",
        default="data/live_trades.db",
        help="Path to SQLite trades database (default: data/live_trades.db)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_report(db_path=args.db_path)

