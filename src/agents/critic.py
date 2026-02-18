"""
src/agents/critic.py

The Behavioral Critic — Bias Hunter / Adversarial Agent.

Its sole job is to find reasons NOT to take a trade that the orchestrator
marked as READY. It specifically hunts for:
  - YES/NO asymmetry exploitation being incorrectly applied
  - Sample size issues masking false edges
  - Market type mismatches (season wins vs game winner vs player prop)
  - Suspicious data patterns (win rate too perfect = data contamination)
  - Kelly fraction sanity checks
  - Recency concerns

Only READY decisions get sent to the critic.
Output: APPROVE or VETO with a specific reason.
"""

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are an adversarial trading critic for a Kalshi prediction market bot.
Your ONLY job is to find legitimate reasons to VETO a trade.

You are the last line of defense before real money is placed.
You are naturally skeptical. You assume the edge might be fake until proven otherwise.

You will receive:
- The original trade signal (ticker, price, action)
- The orchestrator's decision (READY with confidence/edge/kelly)
- The quant's full report (historical data, calibration gap, sample size)

You are specifically hunting for these failure modes:

1. YES/NO ASYMMETRY MISAPPLICATION
   - YES longshots (price <= 20c) are overpriced by up to 64pp vs NO longshots
   - If we're BET_NO on a YES longshot, that's correct — fading the optimism tax
   - If we're BET_YES on a NO longshot (price >= 80c), verify the asymmetry still holds
   - Flag if the yes_no_asymmetry value doesn't support the direction of the trade

2. SUSPICIOUS DATA PATTERNS
   - actual_win_rate of exactly 1.0 or 0.0 = almost certainly data contamination
   - calibration_gap > 0.50 = implausibly large, likely bad data
   - sample_size < 100 with EDGE_CONFIRMED verdict = overfitted

3. MARKET TYPE MISMATCH
   - Season win total markets (KXNBAWINS) behave differently than game winners (KXNBAGAME)
   - Player props (KXNBASGPROP) have different bias profiles than game outcomes
   - Flag if the historical data was aggregated across mixed market types

4. KELLY FRACTION CONCERNS
   - kelly_fraction > 0.10 on MEDIUM confidence = overbetting
   - kelly_fraction > 0.15 on any confidence = always flag

5. RECENCY / REGIME CHANGE
   - If sample spans multiple seasons, early seasons may not reflect current NBA dynamics
   - Flag if we can't confirm the edge holds in recent data

6. LIQUIDITY TRAP
   - volume=0 or open_interest < 500 = can't get filled at this price
   - Even with edge, illiquid markets are a VETO

Respond ONLY with a JSON object — no extra text:
{
    "decision":      "APPROVE" or "VETO",
    "veto_reason":   "<specific reason if VETO, null if APPROVE>",
    "concerns":      ["<list of concerns even if approving — can be empty>"],
    "risk_score":    <int 1-10, where 10 = maximum risk>,
    "summary":       "<one sentence overall assessment>"
}

Be specific in veto_reason. "Insufficient data" is not specific enough.
"actual_win_rate of 1.0 across 7850 samples suggests data contamination,
not genuine edge" is specific enough.

You should APPROVE more often than you VETO — only hard block on clear issues.
"""


class CriticAgent:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0.2,  # Slightly higher than 0 — we want some creative skepticism
        )

    def review(self, trade_packet: dict, orchestrator_decision: dict) -> dict:
        """
        Reviews an orchestrator READY decision and either APPROVEs or VETOs it.
        Only call this when decision['status'] == 'READY'.

        Returns the final decision dict with critic fields added.
        """
        ticker       = trade_packet.get("ticker")
        price        = trade_packet.get("market_price")
        action       = orchestrator_decision.get("action")
        quant        = orchestrator_decision.get("quant_summary", {})
        confidence   = orchestrator_decision.get("confidence")
        edge         = orchestrator_decision.get("edge")
        kelly        = orchestrator_decision.get("kelly_fraction")

        # Detect market type from ticker prefix
        if "KXNBAWINS" in ticker:
            market_type_context = "SEASON WIN TOTAL — different bias profile than game markets"
        elif "KXNBASGPROP" in ticker:
            market_type_context = "PLAYER PROP — different bias profile than game markets"
        else:
            market_type_context = "GAME WINNER — standard longshot bias applies"

        human_msg = f"""Trade to review:
Ticker:         {ticker}
Market Type:    {market_type_context}
Price:          {price}c
Action:         {action}
Confidence:     {confidence}
Edge:           {edge}
Kelly Fraction: {kelly}

Orchestrator Decision:
{json.dumps(orchestrator_decision, indent=2)}

Quant Report:
{json.dumps(quant, indent=2)}

Find reasons to VETO this trade. Be specific.
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw    = response.content
            clean  = re.sub(r"```json|```", "", raw).strip()
            critique = json.loads(clean)

        except Exception as e:
            critique = {
                "decision":    "VETO",
                "veto_reason": f"Critic agent error — defaulting to VETO for safety: {str(e)}",
                "concerns":    [],
                "risk_score":  10,
                "summary":     "Critic failed to run — trade blocked for safety.",
            }

        # Merge critic result into the decision object
        final_status = "APPROVED" if critique["decision"] == "APPROVE" else "VETOED"

        return {
            **orchestrator_decision,
            "status":       final_status,
            "action":       orchestrator_decision["action"] if final_status == "APPROVED" else "PASS",
            "critic": {
                "decision":    critique["decision"],
                "veto_reason": critique.get("veto_reason"),
                "concerns":    critique.get("concerns", []),
                "risk_score":  critique.get("risk_score"),
                "summary":     critique.get("summary"),
            }
        }