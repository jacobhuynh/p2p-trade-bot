"""
src/backtest.py

CLI backtest entrypoint.  Loads finalized Kalshi markets from local parquet,
runs the full agent pipeline (bouncer → quant → orchestrator → critic), and
settles each APPROVED trade immediately using the known market result.

Usage:
    python -m src.backtest
    python -m src.backtest --n 2000 --bankroll 1000 --stake 10
    python -m src.backtest --n 5000 --bankroll 5000 --stake 50 --data-dir /tmp/bt

Environment:
    LLM_MODE      defaults to "rule" (no API keys needed)
    EXECUTION_MODE is forced to "backtest" so the orchestrator skips its
                  own paper-execution step; PaperBroker settles instead.
"""

# ── Set env vars BEFORE any src import so src.config reads the right values ──
import os
os.environ.setdefault("LLM_MODE", "rule")
os.environ["EXECUTION_MODE"] = "backtest"

# ── Standard-library imports ──────────────────────────────────────────────────
import argparse
import contextlib
import io
import sys
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import duckdb
except ImportError:
    print("[ERROR] duckdb not installed.  Run: pip install duckdb")
    sys.exit(1)

# ── Project imports (after env vars are set) ──────────────────────────────────
from src.pipeline.bouncer import process_trade
from src.agents.orchestrator import LeadAnalyst
from src.execution.trade_manager import PaperBroker

# ── DuckDB glob for the parquet files ─────────────────────────────────────────
_MARKETS_GLOB = str(Path("data/kalshi/markets/*.parquet"))


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_signals(n: int) -> list:
    """
    Load up to `n` finalized markets ordered by close_time ascending.

    Returns
    -------
    list of dicts with keys: ticker, last_price (int), result ("yes"/"no")
    """
    con = duckdb.connect()
    try:
        df = con.execute(f"""
            SELECT ticker, CAST(last_price AS INTEGER) AS last_price, result
            FROM read_parquet('{_MARKETS_GLOB}')
            WHERE status    = 'finalized'
              AND result    IN ('yes', 'no')
              AND last_price BETWEEN 1 AND 99
            ORDER BY close_time ASC
            LIMIT {n}
        """).df()
    except Exception as exc:
        print(f"[ERROR] Cannot read parquet data: {exc}")
        print("        Generate the mock dataset first:")
        print("          python3 mock_database_setup.py")
        sys.exit(1)

    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Core backtest loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(n: int, bankroll: float, stake: float, data_dir: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  Backtest  |  LLM_MODE=rule  |  EXECUTION_MODE=backtest")
    print(f"  n={n:,}  bankroll=${bankroll:.2f}  stake=${stake:.2f}/trade")
    print(f"{'='*62}\n")

    signals = load_signals(n)
    print(f"  Loaded {len(signals):,} finalized markets from parquet")

    # ── Diagnostic: how many longshot markets did the generator produce? ──────
    n_ls_no  = sum(1 for r in signals if r["last_price"] <= 20)
    n_ls_yes = sum(1 for r in signals if r["last_price"] >= 80)
    print(f"  Longshot YES (price ≤20c, BET_NO signals) : {n_ls_no:>5,}")
    print(f"  Longshot NO  (price ≥80c, BET_YES signals): {n_ls_yes:>5,}")
    if n_ls_no == 0 and n_ls_yes == 0:
        print(
            "\n  [WARN] No longshot markets found — "
            "run mock_database_setup.py to regenerate data."
        )
    print()

    analyst = LeadAnalyst()
    broker  = PaperBroker(bankroll=bankroll, stake=stake, data_dir=data_dir)

    n_processed = 0
    n_bounced   = 0   # rejected by bouncer (not longshot)
    n_passed    = 0   # PASS from orchestrator/quant
    n_vetoed    = 0   # VETOED by critic
    n_approved  = 0   # APPROVED → sent to broker
    n_filled    = 0   # actually settled

    for row in signals:
        ticker    = row["ticker"]
        yes_price = int(row["last_price"])
        result    = row["result"]
        n_processed += 1

        # ── Bouncer ──────────────────────────────────────────────────────────
        # Suppress the bouncer's per-skip print lines to keep output readable.
        trade_data = {"market_ticker": ticker, "yes_price": yes_price}
        with contextlib.redirect_stdout(io.StringIO()):
            trade_packet = process_trade(trade_data)

        if trade_packet is None:
            n_bounced += 1
            continue

        # ── Full agent pipeline ───────────────────────────────────────────────
        # EXECUTION_MODE=backtest → orchestrator skips its paper step.
        decision = analyst.analyze_signal(trade_packet)
        status   = decision.get("status")

        if status == "PASS":
            n_passed += 1
            continue
        if status == "VETOED":
            n_vetoed += 1
            continue
        if status != "APPROVED":
            continue

        n_approved += 1

        # ── Immediate settlement via PaperBroker ─────────────────────────────
        report = broker.paper_execute(decision, result)
        if report.get("status") == "SETTLED":
            n_filled += 1
            sign = "+" if report["pnl"] >= 0 else ""
            print(
                f"  [{n_filled:>4}] {ticker:<42} "
                f"{decision['action']:<7}  {yes_price:>3}c  "
                f"result={result}  "
                f"pnl={sign}{report['pnl']:.2f}  "
                f"cash={report['cash']:.2f}"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    summary = broker.summary()

    print(f"\n{'='*62}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*62}")
    print(f"  Markets processed  : {n_processed:>7,}")
    print(f"  Bouncer rejected   : {n_bounced:>7,}  (not longshot)")
    print(f"  Pipeline PASS      : {n_passed:>7,}")
    print(f"  Pipeline VETOED    : {n_vetoed:>7,}")
    print(f"  APPROVED / settled : {n_approved:>7,} / {n_filled:>6,}")
    print(f"  {'─'*54}")
    print(f"  Bankroll           : ${summary['bankroll']:>10.2f}")
    print(f"  Final cash         : ${summary['final_cash']:>10.2f}")
    print(f"  Total PnL          : ${summary['total_pnl']:>+10.2f}")
    print(f"  ROI                : {summary['roi']*100:>+9.2f}%")
    win_str = f"{summary['n_wins']}/{summary['n_trades']}"
    print(f"  Win rate           : {summary['win_rate']*100:>9.1f}%  ({win_str})")
    print(f"  Max drawdown       : ${summary['max_drawdown']:>10.2f}")
    print(f"{'='*62}")
    print(f"\n  Output files (in {data_dir}/):")
    print(f"    backtest_trades.csv   — per-trade log")
    print(f"    backtest_equity.csv   — equity curve")
    print(f"    backtest_book.json    — summary stats")

    if n_filled == 0:
        print(
            "\n  NOTE: 0 trades were approved.\n"
            "  If you have not yet regenerated the parquet data, run:\n"
            "    python3 mock_database_setup.py\n"
            "  The mock dataset must be regenerated after the longshot-bias\n"
            "  patch to mock_database_setup.py for edges to appear."
        )

    print()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the p2p-trade-bot pipeline against historical Kalshi data. "
            "Requires data/kalshi/ parquet files (run mock_database_setup.py first)."
        )
    )
    parser.add_argument(
        "--n", type=int, default=2000,
        help="Number of finalized markets to process (default: 2000)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0,
        help="Starting cash in dollars (default: 1000.0)",
    )
    parser.add_argument(
        "--stake", type=float, default=10.0,
        help="Fixed stake per trade in dollars (default: 10.0)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/paper",
        help="Output directory for backtest files (default: data/paper)",
    )
    args = parser.parse_args()

    run_backtest(
        n        = args.n,
        bankroll = args.bankroll,
        stake    = args.stake,
        data_dir = args.data_dir,
    )


if __name__ == "__main__":
    main()
