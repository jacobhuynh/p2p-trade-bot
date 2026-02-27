"""
src/agents/quant.py

The Quant Agent — Historical Edge / Calibration Gap Analyzer.

Key insight: we never query by ticker (live tickers won't exist in historical DB).
Instead we query by PRICE BUCKET across all finalized NBA markets.
The question is: "Historically, what actually happened to NBA contracts priced at X cents?"

All math (calibration gap, verdict, implied probability) is computed in Python
BEFORE calling the LLM.  The LLM's only job is to write a qualitative one-sentence
summary that incorporates the pre-computed numbers and any available live context
(ESPN game status, recent team records from nba_api).

Real prediction market edges are small:
  - 1.5%+ calibration gap = strong edge
  - 0.75%+ calibration gap = weak but real
  - <0.75% = noise, not tradeable
"""

import json
import re

from src.tools.duckdb_tool import (
    get_historical_win_rate,
    get_longshot_bias_stats,
    get_price_bucket_edge,
    get_market_volume_stats,
)


def _contracts_at_price(orderbook: dict | None, yes_price: int, action: str) -> int | None:
    """
    Return the number of contracts available in the live order book at our target price.

    Three-state return:
      None  — orderbook is None (offline / creds missing / network error) → depth unknown
      0     — orderbook available but our target price has no contracts → confirmed empty
      N > 0 — N contracts available at our price → can fill up to N contracts

    For BET_NO: we buy NO contracts. The order book's "no" side lists available NO
    contracts at each no_price (= 100 - yes_price).
    For BET_YES: we buy YES contracts. The "yes" side lists available YES contracts.

    Kalshi order book format: {"yes": [[price, size], ...], "no": [[price, size], ...]}
    """
    if orderbook is None:
        return None

    if action == "BET_NO":
        target_price = 100 - yes_price
        entries = orderbook.get("no", [])
    else:
        target_price = yes_price
        entries = orderbook.get("yes", [])

    for entry in entries:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            if int(entry[0]) == target_price:
                return int(entry[1])

    return 0  # price not found in book — confirmed zero depth

_SYSTEM_PROMPT = """You are a quantitative analyst for a Kalshi prediction market trading bot.

All numerical calculations (calibration gap, implied probability, verdict) have already
been computed in Python and are provided to you as verified facts.  Do NOT recompute them.

Your ONLY task is to write a single-sentence qualitative summary that:
  - States the calibration gap and verdict in plain English
  - Integrates any live game context (ESPN status, injury flags) if available
  - Notes team momentum from recent records if available
  - Flags any data quality concerns (low sample, perfect win rate, etc.)

Respond ONLY with a JSON object — no extra text:
{"summary": "<one sentence>"}
"""


class QuantAgent:
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
        )

    def analyze(self, trade_packet: dict) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")
        ticker = trade_packet.get("ticker", "")

        # ── Query by PRICE BUCKET, not ticker ─────────────────────────────────
        edge_data    = get_price_bucket_edge(price, action)
        bias_data    = get_longshot_bias_stats(price)
        win_data     = get_historical_win_rate(price)
        inverse_edge = get_price_bucket_edge(100 - price, "BET_YES" if action == "BET_NO" else "BET_NO")

        # ── Compute all math in Python ─────────────────────────────────────────
        implied_prob    = round((100 - price) / 100, 4) if action == "BET_NO" else round(price / 100, 4)
        calibration_gap = edge_data.get("edge")           # already Python-computed by duckdb_tool
        actual_win_rate = edge_data.get("actual_win_rate")
        sample_size     = edge_data.get("sample_size", 0)

        # yes_no_asymmetry: difference between our side edge and the inverse side edge
        inv_edge_val = inverse_edge.get("edge")
        yes_no_asymmetry = (
            round(calibration_gap - inv_edge_val, 4)
            if calibration_gap is not None and inv_edge_val is not None
            else None
        )

        # Verdict (mirrors previous LLM rules, now enforced in Python)
        if sample_size < 100 or calibration_gap is None:
            verdict      = "INSUFFICIENT_DATA"
            data_quality = "INSUFFICIENT"
        elif calibration_gap > 0.015 and sample_size >= 200:
            verdict      = "EDGE_CONFIRMED"
            data_quality = "SUFFICIENT"
        elif calibration_gap > 0.0075 and sample_size >= 100:
            verdict      = "EDGE_WEAK"
            data_quality = "SUFFICIENT"
        else:
            verdict      = "NO_EDGE"
            data_quality = "SUFFICIENT"

        # ── Fetch live context (graceful — never blocks the pipeline) ──────────
        game_context: dict | None = None
        team_stats:   dict | None = None
        orderbook_depth_at_price: int | None = None

        try:
            from src.tools.espn_tool import find_game
            game_context = find_game(ticker)
        except Exception:
            pass

        try:
            from src.tools.nba_tool import get_team_recent_records
            team_stats = get_team_recent_records(ticker)
        except Exception:
            pass

        try:
            from src.tools.kalshi_rest import get_orderbook
            orderbook = get_orderbook(ticker)
            orderbook_depth_at_price = _contracts_at_price(orderbook, price, action)
        except Exception:
            pass

        # ── Ask LLM for qualitative summary only ──────────────────────────────
        human_msg = f"""Pre-computed Analysis:
Ticker:          {ticker}
Price:           {price}c
Action:          {action}
Implied Prob:    {implied_prob} ({implied_prob*100:.1f}%)
Calibration Gap: {calibration_gap}  (actual_win_rate - implied_prob)
Actual Win Rate: {actual_win_rate}
Sample Size:     {sample_size}
Verdict:         {verdict}
Data Quality:    {data_quality}
Yes/No Asymmetry:{yes_no_asymmetry}
No Win Rate:     {bias_data.get('no_win_rate')} (longshot bias stat)

Live ESPN Context:
{json.dumps(game_context, indent=2) if game_context else "No game found for today/yesterday."}

Team Recent Records (nba_api):
{json.dumps(team_stats, indent=2) if team_stats else "nba_api data unavailable."}

Write a single-sentence qualitative summary.  Use the pre-computed values — do not recalculate.
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw   = response.content
            clean = re.sub(r"```json|```", "", raw).strip()
            llm_out = json.loads(clean)
            summary = llm_out.get("summary", "")
        except Exception as e:
            summary = f"Quant summary unavailable: {str(e)}"

        # ── Assemble and return — same keys as before so downstream is unchanged ─
        return {
            "historical_edge":          calibration_gap,
            "actual_win_rate":          actual_win_rate,
            "implied_prob":             implied_prob,
            "no_win_rate":              bias_data.get("no_win_rate"),
            "yes_no_asymmetry":         yes_no_asymmetry,
            "sample_size":              sample_size,
            "data_quality":             data_quality,
            "verdict":                  verdict,
            "calibration_gap":          calibration_gap,
            "summary":                  summary,
            "game_context":             game_context,
            "team_stats":               team_stats,
            # ── Live liquidity ──────────────────────────────────────────────────
            # orderbook_depth_at_price: None = unknown (offline), 0 = confirmed empty, N = available
            "orderbook_depth_at_price": orderbook_depth_at_price,
            # ── Raw query results — separate DuckDB queries, different population cuts ──
            "price_bucket_edge":        edge_data,    # get_price_bucket_edge(price, action)
            "longshot_bias":            bias_data,    # get_longshot_bias_stats(price)
            "taker_win_rate":           win_data,     # get_historical_win_rate(price)
            "inverse_bucket":           inverse_edge, # get_price_bucket_edge(100-price, opposite)
        }
