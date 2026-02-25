"""
src/settle.py

Resolution checker for the live mock trader.

For each PENDING_RESOLUTION trade in data/live_trades.db, queries the Kalshi
REST API to check whether the market has been finalized.  When Kalshi reports
status="finalized" the result field ("yes" or "no") is used directly to
evaluate the mock trade and compute hypothetical P&L.

Works for all market types (KXNBAGAME, KXNBAWINS, KXNBASGPROP) — no ticker
parsing or team-name inference required.

Usage:
    python -m src.settle
"""

from src.execution.trade_logger import TradeLogger
from src.tools.kalshi_rest import get_market_details


def run_settle(db_path: str = "data/live_trades.db") -> None:
    logger = TradeLogger(db_path=db_path)
    pending_trades = logger.open_trades()   # PENDING_RESOLUTION rows

    if not pending_trades:
        print("  No pending trades to evaluate.")
        return

    print(f"\n{'='*66}")
    print(f"  Resolution check  |  {len(pending_trades)} pending trade(s)")
    print(f"{'='*66}")

    newly_evaluated = []
    still_pending   = []

    for trade in pending_trades:
        ticker = trade["ticker"]

        market = get_market_details(ticker)

        if market is None:
            still_pending.append(trade)
            print(f"  [#{trade['id']:>3}] {ticker:<42}  Kalshi API unavailable — kept PENDING")
            continue

        status = market.get("status", "")
        if status != "finalized":
            still_pending.append(trade)
            print(f"  [#{trade['id']:>3}] {ticker:<42}  status={status} — kept PENDING")
            continue

        result = market.get("result", "")
        if result not in ("yes", "no"):
            still_pending.append(trade)
            print(f"  [#{trade['id']:>3}] {ticker:<42}  result={result!r} unrecognised — kept PENDING")
            continue

        evaluated = logger.evaluate_trade(trade["id"], result)
        newly_evaluated.append({**trade, **evaluated})
        sign    = "+" if evaluated["pnl_usd"] >= 0 else ""
        outcome = "WIN" if evaluated["won"] else "LOSS"
        print(
            f"  [#{trade['id']:>3}] {ticker:<42}  "
            f"{trade['action']:<7}  {trade['yes_price']:>3}c  "
            f"result={result}  {outcome}  pnl={sign}${evaluated['pnl_usd']:.2f}"
        )

    print(f"\n  {'─'*62}")
    print(f"  Newly evaluated : {len(newly_evaluated)}")
    print(f"  Still pending   : {len(still_pending)}")

    if newly_evaluated:
        summary = logger.summary()
        print(f"\n  Cumulative P&L on all evaluated trades:")
        print(f"    Trades     : {summary['n_trades']}")
        print(f"    Win rate   : {summary['win_rate']*100:.1f}%  ({summary['n_wins']}/{summary['n_trades']})")
        print(f"    Total P&L  : ${summary['total_pnl']:+.2f}")
        print(f"    ROI        : {summary['roi']*100:+.1f}%")

    print(f"{'='*66}\n")


if __name__ == "__main__":
    run_settle()
