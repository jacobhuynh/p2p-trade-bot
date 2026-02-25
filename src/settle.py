"""
src/settle.py

Resolution checker for the live mock trader.

For each PENDING_RESOLUTION trade in data/live_trades.db, queries the ESPN
hidden scoreboard API to check if the game has finished.  When ESPN reports
STATUS_FINAL, evaluates the mock trade and computes the hypothetical P&L,
updating the row to EVALUATED.

No Kalshi API calls are made — resolution is driven entirely by ESPN.

Usage:
    python -m src.settle
"""

from src.execution.trade_logger import TradeLogger
from src.tools.espn_tool import find_game


def _determine_result(trade: dict, game: dict) -> str | None:
    """
    Given an EVALUATED trade and an ESPN final game result, return 'yes' or 'no'
    (the Kalshi market result) based on which team won.

    For KXNBAGAME markets:
      - The ticker encodes HOME and AWAY teams (e.g. KXNBAGAME-25JAN15LACBOS-NO)
      - YES resolves 'yes' if the YES side won the game
      - The trade's 'side' field ('yes' or 'no') tells us which Kalshi side we hold
      - We need to figure out whether 'yes' = home win or 'yes' = away win

    Kalshi convention: the ticker suffix (-YES / -NO) tells us what YES represents.
    For simplicity we use the winner_abbr from ESPN and the home/away teams parsed
    from the ticker to decide: if winner is home_abbr → 'yes' (home won → YES wins
    for a standard KXNBAGAME market). If winner is away_abbr → 'no'.

    Returns None if the result cannot be determined (draw / OT ambiguity / parse fail).
    """
    winner = game.get("winner_abbr", "").upper()
    if not winner:
        return None

    home_espn = game.get("home_abbr", "").upper()
    away_espn = game.get("away_abbr", "").upper()

    if winner == home_espn:
        return "yes"   # home team won → YES resolves 'yes'
    if winner == away_espn:
        return "no"    # away team won → YES resolves 'no'
    return None


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

        game = find_game(ticker, search_days=3)

        if game is None:
            still_pending.append(trade)
            print(f"  [#{trade['id']:>3}] {ticker:<42}  No ESPN match — kept PENDING")
            continue

        if game["status"] != "STATUS_FINAL":
            still_pending.append(trade)
            status_label = game.get("status", "UNKNOWN")
            print(f"  [#{trade['id']:>3}] {ticker:<42}  Game {status_label} — kept PENDING")
            continue

        result = _determine_result(trade, game)
        if result is None:
            still_pending.append(trade)
            print(f"  [#{trade['id']:>3}] {ticker:<42}  Cannot determine result — kept PENDING")
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
