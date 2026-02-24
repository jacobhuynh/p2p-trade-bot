# p2p-trade-bot

Kalshi prediction-market trading bot that identifies and evaluates NBA longshot-bias
opportunities via a multi-agent pipeline:

```
WebSocket stream → Bouncer → QuantAgent → Orchestrator → CriticAgent → TradeManager
```

---

### setup

```bash
pip install -r requirements.txt
python3 mock_database_setup.py   # generates mock parquet data under data/kalshi/
```

---

### LLM mode

Control which agent implementation is used via `LLM_MODE`:

| Value | Behaviour |
|---|---|
| `rule` **(default)** | Deterministic rule-based agents — zero external API calls, no keys required |
| `anthropic` | ChatAnthropic (Claude Sonnet 4.6) — requires `ANTHROPIC_API_KEY` |

---

### Execution mode

Control what happens after an **APPROVED** decision via `EXECUTION_MODE`:

| Value | Behaviour |
|---|---|
| `paper` **(default)** | Simulate the trade and record PnL to local files — no real money |
| `live` | Raises `NotImplementedError` (placeholder for real Kalshi order execution) |
| `backtest` | Set automatically by `src/backtest.py` — orchestrator skips execution; `PaperBroker` settles instead |

---

### Paper trading files

All paper trading state is stored under `data/paper/` (override with `PAPER_DATA_DIR`):

| File | Contents |
|---|---|
| `book.json` | Current portfolio: cash balance, open positions, realized PnL |
| `trades.csv` | Append-only log of every simulated fill |
| `equity.csv` | Append-only equity-curve snapshot after every trade |

**book.json schema**
```json
{
  "cash": 985.14,
  "positions": {
    "KXNBAGAME-26FEB19BKNCLE-BKN": {
      "side": "NO",
      "contracts": 2,
      "avg_price": 86.0
    }
  },
  "realized_pnl": 0.0,
  "updated_at": "2026-02-23T16:00:00+00:00"
}
```

**trades.csv columns**
`timestamp, market_ticker, side, action, price, contracts, notional, cash_after, note`

**equity.csv columns**
`timestamp, cash, unrealized_pnl, realized_pnl, equity, n_positions`

---

### Run commands

```bash
# Default — rule agents + paper trading, zero API keys needed
python3 tests/test_pipeline.py

# Same via pytest
pytest tests/ -v -s

# Rule agents + paper trading, custom starting cash
PAPER_STARTING_CASH=5000 python3 tests/test_pipeline.py

# Anthropic LLM agents + paper trading (requires ANTHROPIC_API_KEY)
LLM_MODE=anthropic python3 tests/test_pipeline.py --live

# Live pipeline with real DB data (rule mode, paper execution)
python3 tests/test_pipeline.py --live
```

`.env` example:
```
# Agent mode
LLM_MODE=anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Execution mode
EXECUTION_MODE=paper
PAPER_STARTING_CASH=1000.0
PAPER_MAX_CONTRACTS=20
PAPER_DATA_DIR=data/paper

# Kalshi (optional — only needed for live WebSocket trading)
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=path/to/key.pem
```

> **Kalshi REST keys are optional.** When absent, market-enrichment fields in
> the trade packet are set to `"Unknown"` and the rest of the pipeline runs
> normally.

---

### Backtest

Replay historical finalized markets through the full agent pipeline and compute
actual PnL, win rate, ROI, and max drawdown.

```bash
# Default — processes 2000 markets, $1000 bankroll, $10 fixed stake
python -m src.backtest

# Custom parameters
python -m src.backtest --n 5000 --bankroll 5000 --stake 50

# Write output files to a custom directory
python -m src.backtest --n 2000 --bankroll 1000 --stake 10 --data-dir /tmp/bt
```

| Flag | Default | Description |
|---|---|---|
| `--n` | `2000` | Number of finalized markets to process |
| `--bankroll` | `1000.0` | Starting cash in dollars |
| `--stake` | `10.0` | Fixed dollar amount risked per trade |
| `--data-dir` | `data/paper` | Output directory |

**Output files** (written fresh on each run, not appended):

| File | Contents |
|---|---|
| `backtest_trades.csv` | Per-trade log: action, side, price, qty, cost, result, payout, pnl |
| `backtest_equity.csv` | Cash balance after every settled trade |
| `backtest_book.json` | Final summary: bankroll, total\_pnl, win\_rate, ROI, max\_drawdown |

**Settlement convention**

Each APPROVED trade is settled immediately using the known market `result`:

| Action | Entry price | Wins when |
|---|---|---|
| `BET_NO` | `(100 - yes_price)` cents / contract | `result == "no"` |
| `BET_YES` | `yes_price` cents / contract | `result == "yes"` |

Payout is always **$1.00 per contract** on a win, **$0** on a loss.

> **Note on mock data:** the generated dataset has no systematic longshot bias
> (market prices are close to true probabilities), so 0 approved trades is
> expected.  Real Kalshi data will surface genuine edges.

---

### Running the live bot

```bash
python -m src.pipeline.websocket_client
```

Requires Kalshi WebSocket credentials in `.env`.
