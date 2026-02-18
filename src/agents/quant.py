"""
src/agents/quant.py

The Quant Agent — Historical Edge / Calibration Gap Analyzer.

Key insight: we never query by ticker (live tickers won't exist in historical DB).
Instead we query by PRICE BUCKET across all finalized NBA markets.
The question is: "Historically, what actually happened to NBA contracts priced at X cents?"

Real prediction market edges are small:
  - 2%+ calibration gap = strong edge
  - 0.8%+ calibration gap = weak but real
  - <0.8% = noise, not tradeable
"""

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.tools.duckdb_tool import (
    get_historical_win_rate,
    get_longshot_bias_stats,
    get_price_bucket_edge,
    get_market_volume_stats,
)

SYSTEM_PROMPT = """You are a quantitative analyst for a Kalshi prediction market trading bot.
Your job is to identify "Calibration Gaps" — cases where the market price is statistically
mispriced compared to historical outcomes across all NBA contracts at that price.

IMPORTANT: The data you receive is aggregated across ALL historical NBA contracts at this
price point — not specific to the current ticker. This is intentional.

IMPORTANT: Real prediction market edges are very small. Do not expect large edges.
  - calibration_gap > 0.02 (2%) = strong edge, historically rare
  - calibration_gap > 0.008 (0.8%) = weak but real edge
  - calibration_gap <= 0.008 = noise, not a tradeable edge

The calibration_gap is the ONLY edge metric that matters. Calculate it as:
  calibration_gap = actual_win_rate_for_our_side - implied_probability_for_our_side

For BET_NO at price X:
  implied_probability = (100 - X) / 100
  actual_win_rate = how often NO actually won at this price historically

For BET_YES at price X:
  implied_probability = X / 100
  actual_win_rate = how often YES actually won at this price historically

Respond ONLY with a JSON object — no extra text:
{
    "historical_edge":    <float or null>,
    "actual_win_rate":    <float or null>,
    "implied_prob":       <float or null>,
    "no_win_rate":        <float or null>,
    "yes_no_asymmetry":   <float or null>,
    "sample_size":        <int>,
    "data_quality":       "SUFFICIENT" or "INSUFFICIENT",
    "verdict":            "EDGE_CONFIRMED" or "EDGE_WEAK" or "NO_EDGE" or "INSUFFICIENT_DATA",
    "calibration_gap":    <float or null>,
    "summary":            "<one sentence explaining the calibration gap finding>"
}

Verdict rules (calibrated to real market edge sizes):
- EDGE_CONFIRMED:    calibration_gap > 0.02  AND sample_size >= 200
- EDGE_WEAK:         calibration_gap > 0.008 AND sample_size >= 100
- NO_EDGE:           calibration_gap <= 0.008
- INSUFFICIENT_DATA: sample_size < 100 or data unavailable

Never use the yes_price itself as the edge value. The edge is always the
difference between actual outcomes and implied probability.
"""


class QuantAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0,
        )

    def analyze(self, trade_packet: dict) -> dict:
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")

        # ── Query by PRICE BUCKET, not ticker ─────────────────────────
        edge_data    = get_price_bucket_edge(price, action)
        bias_data    = get_longshot_bias_stats(price)
        win_data     = get_historical_win_rate(price)
        inverse_edge = get_price_bucket_edge(100 - price, "BET_YES" if action == "BET_NO" else "BET_NO")

        # Compute implied prob for context
        if action == "BET_NO":
            implied_prob = round((100 - price) / 100, 4)
        else:
            implied_prob = round(price / 100, 4)

        human_msg = f"""Trade Signal:
Price:        {price}c
Action:       {action}
Implied prob for our side: {implied_prob} ({implied_prob*100:.1f}%)

Price Bucket Analysis (all NBA contracts historically priced at {price}c):
  Edge Data:          {json.dumps(edge_data,    indent=2)}
  Longshot Bias:      {json.dumps(bias_data,    indent=2)}
  Taker Win Rate:     {json.dumps(win_data,     indent=2)}

Inverse Side ({100 - price}c):
  Edge Data:          {json.dumps(inverse_edge, indent=2)}

Calculate calibration_gap = actual_win_rate_for_our_side - {implied_prob}
Do NOT use {price/100} as the edge. Use the difference from historical outcomes only.
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw   = response.content
            clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(clean)

        except Exception as e:
            return {
                "historical_edge":  None,
                "actual_win_rate":  None,
                "implied_prob":     None,
                "no_win_rate":      None,
                "yes_no_asymmetry": None,
                "sample_size":      0,
                "data_quality":     "INSUFFICIENT",
                "verdict":          "INSUFFICIENT_DATA",
                "calibration_gap":  None,
                "summary":          f"Quant agent error: {str(e)}",
            }