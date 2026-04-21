# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Generate mock historical parquet data (required once before running)
python mock_database_setup.py

# Stream live Kalshi trades through the full agent pipeline
python -m src.pipeline.websocket_client

# Resolve PENDING trades via Kalshi REST, write P&L to SQLite
python -m src.settle

# Summary report of evaluated trades, bankroll, markets
python -m src.report_trades

# Full test suite excluding live websocket (no API keys required)
pytest tests/ --ignore=tests/test_websocket.py -v

# Single test file / single test
pytest tests/test_bouncer.py -v
pytest tests/test_pipeline.py::test_pipeline_approved -v

# WebSocket integration test (requires KALSHI_* + ANTHROPIC_API_KEY in .env)
pytest tests/test_websocket.py -v

# Full pipeline with real Claude + real parquet data
python tests/test_pipeline.py --live

# Manual sentiment/ESPN tool check
python scripts/verify_sentiment.py
```

## Architecture

**Trading thesis:** Kalshi retail overprices YES underdogs on NBA game-winner markets. At a YES price of 14¢ the team may win only ~8% of the time — a calibration gap that BET_NO exploits over many trades. The bot measures that gap from historical parquet data and trades when it exceeds a threshold.

**Data flow:**

```
WebSocket → Router → Bouncer → (Quant ∥ Sentiment) → Orchestrator → Critic → TradeLogger
                                                                              ↓
                                                              src/settle.py ← Kalshi REST
```

### Key cross-file invariants

- **All math is computed in Python, never in the LLM.** `QuantAgent` queries DuckDB, computes `calibration_gap = actual_win_rate − implied_probability`, and only asks Claude for a one-sentence qualitative summary. Any change that pushes numerical reasoning into a prompt violates the design.
- **Historical queries bucket by price, not ticker.** Live tickers won't appear in historical parquet. `duckdb_tool.py` aggregates across all finalized NBA markets at each price bucket. Don't write queries keyed on the live ticker.
- **Longshot thresholds are hard-coded in `bouncer.py`:** YES ≤ 20¢ → `BET_NO`, YES ≥ 80¢ → `BET_YES`, mid-price dropped. These are the only trades that reach the agent stack.
- **Router dispatch is prefix-based.** `KXNBAGAME-*` goes through the full pipeline; `KXNBAWINS-*` (totals) and `KXNBASGPROP-*` (player props) are placeholders that print and return. Everything else is silently dropped. Adding a new market type means wiring both `router.py` classification and a handler.
- **Kelly fraction is capped at 15%** in `orchestrator.py`; `PaperTradeManager` additionally caps `risk_fraction` at 2% of cash per trade. Both caps exist intentionally — don't remove one without the other.

### Agent model assignments (don't swap casually)

| Agent          | Model                | Rationale                                        |
| -------------- | -------------------- | ------------------------------------------------ |
| SentimentAgent | `claude-haiku-4-5`   | Cheap summary of ESPN news context               |
| Orchestrator   | `claude-haiku-4-5`   | Merges quant + sentiment into short narrative    |
| CriticAgent    | `claude-sonnet-4-6`  | Primary decision-maker; adversarial VETO search  |

### Critic-specific behavior

The Critic's only job is to find reasons to **VETO**. Before invoking the LLM it queries the SQLite trade log for open positions and injects portfolio concentration data into the prompt. It explicitly understands that BET_NO on a cheap underdog is the strategy, not a red flag — don't add prompt language that treats longshot contrarian bets as suspicious.

### Trade lifecycle

1. `TradeLogger.log_trade()` writes row with `status=PENDING_RESOLUTION` to `data/live_trades.db`.
2. `src/settle.py` polls `kalshi_rest.get_market_details(ticker)`; when `status=="finalized"` it reads the `result` field and calls `logger.evaluate_trade(id, result)` → `status=EVALUATED` with `pnl_usd`.
3. Settlement is driven by **Kalshi REST**, not ESPN. ESPN is used only for live in-game context (QuantAgent) and news sentiment (SentimentAgent).

### Graceful degradation expectations

- Missing Kalshi credentials → `get_market_details` returns `None`; bouncer fills `market_title`/`rules_primary` with `"Unknown"` and pipeline continues.
- `nba_tool.get_team_recent_records` returns `None` on timeout/parse failure; callers must not assume success.
- `SentimentAgent` short-circuits non-GAME_WINNER packets (TOTALS, PLAYER_PROP pass through with `sentiment_context=None`).
- `WebSocket run_forever()` uses exponential backoff (1s → 60s cap) and resets on successful reconnect. Don't add a retry layer above it.

### Team abbreviation quirk

`espn_tool.py` maps Kalshi→ESPN abbreviations (`GSW→GS`, `NOP→NO`, `UTA→UTAH`). The ticker parser uses `{2,3}` character matching to split adjacent 3-char team codes (e.g. `LACBOS` → `LAC` + `BOS`). Any new team-code logic must preserve both behaviors.

## Data

- `data/kalshi/{markets,trades}/*.parquet` — historical data. **Mocked by default** via `mock_database_setup.py`. Real data can be dropped in from [jon-becker/prediction-market-analysis](https://github.com/jon-becker/prediction-market-analysis).
- `data/live_trades.db` — SQLite, auto-created. Query with `sqlite3 data/live_trades.db ...`.
- `data/paper/` — `PaperTradeManager` writes `book.json`, `trades.csv`, `equity.csv` here.

## Configuration

`.env` in repo root. `ANTHROPIC_API_KEY` is required for any agent work. `KALSHI_API_KEY_ID` + `KALSHI_PRIVATE_KEY_PATH` are required for live WebSocket streaming and for `settle.py`; absent them the pipeline still runs on mock data with `"Unknown"` enrichment fields. Paper sizing knobs: `PAPER_STARTING_CASH` (default 1000), `PAPER_MAX_CONTRACTS` (default 20).

## Testing notes

Tests hit **real public APIs** (ESPN, nba_api, DuckDB) by default — only LLM and Kalshi REST calls are mocked in the core unit suite. Expect occasional flakes on live ESPN/nba_api; those tests return `None` gracefully rather than failing when the upstream is unavailable. `test_websocket.py` auto-skips when Kalshi/Anthropic credentials are missing.
