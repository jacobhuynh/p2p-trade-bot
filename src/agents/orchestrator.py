"""
src/agents/orchestrator.py

The Lead Analyst — orchestrates the quant agent, makes a preliminary
READY/PASS decision, then sends READY decisions to the critic for
final approval before returning an execution-ready decision object.

Pipeline:
  trade_packet
    → quant  (calibration gap + ESPN context + nba_api stats)
    → [Research Agent placeholder — not yet implemented]
    → orchestrator (synthesize → Trade Proposal READY/PASS)
    → critic (adversarial APPROVE/VETO — only on READY)

Always uses Claude (claude-sonnet-4-6). Requires ANTHROPIC_API_KEY.
"""

import json
import re
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# THRESHOLDS
# Calibrated to real prediction market edge sizes.
# Real edges are small — 2% is strong, 0.8% is weak but real.
# ─────────────────────────────────────────────

MIN_EDGE_HIGH      = 0.02    # 2%+ calibration gap = strong edge
MIN_EDGE_MEDIUM    = 0.008   # 0.8%+ calibration gap = weak but real
MIN_SAMPLE_SIZE    = 200     # need meaningful history to trust the gap
MIN_VOLUME         = 5000    # minimum market volume for liquidity
KELLY_FRACTION_CAP = 0.15


def _kelly(edge: float) -> float:
    if edge <= 0:
        return 0.0
    return round(min(edge, KELLY_FRACTION_CAP), 4)


def _confidence(edge: float, sample_size: int) -> str:
    if sample_size < MIN_SAMPLE_SIZE:
        return "LOW"
    if edge >= MIN_EDGE_HIGH:
        return "HIGH"
    if edge >= MIN_EDGE_MEDIUM:
        return "MEDIUM"
    return "LOW"


_SYSTEM_PROMPT = """You are a lead trading analyst for a Kalshi prediction market trading bot.

You receive:
  1. A trade signal (ticker, price, action)
  2. A quantitative report with pre-computed calibration gap, verdict, and live context
     (ESPN game status, team recent records from nba_api)
  3. A research report — contextual news, injury updates, line movement
     (may be marked as "not yet available" if the Research Agent is not yet implemented)

Your job is to synthesize all available data into a Trade Proposal: READY or PASS.

IMPORTANT: Real prediction market edges are small. A 2% calibration gap is
considered strong. A 0.8% gap is weak but real. Do not expect large edges.

Decision rules:
- READY if: verdict is EDGE_CONFIRMED or EDGE_WEAK, AND confidence is HIGH or MEDIUM
- PASS if: verdict is NO_EDGE or INSUFFICIENT_DATA, or confidence is LOW
- Always PASS if sample_size < {min_sample}
- Always PASS if calibration_gap is null or <= 0
- Live context (injuries, game status) can downgrade READY to PASS if there is a
  clear disqualifying signal (e.g. key player ruled out, game postponed)

Respond ONLY with a JSON object — no extra text:
{{
    "status":  "READY" or "PASS",
    "reason":  "<one sentence justification — cite the key factor>"
}}
""".format(min_sample=MIN_SAMPLE_SIZE)


class LeadAnalyst:
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        from src.agents.quant import QuantAgent
        from src.agents.critic import CriticAgent

        self.role   = "Lead Trading Orchestrator"
        self.quant  = QuantAgent()
        self.critic = CriticAgent()
        self.llm    = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0,
        )

    # ── Internal: orchestrator decision step ────────────────────────────────
    def _orchestrate(
        self,
        quant_report:     dict,
        confidence:       str,
        sample_size:      int,
        calibration_gap,
        trade_packet:     dict,
        research_report:  dict | None = None,
    ) -> dict:
        """Returns {"status": "READY"/"PASS", "reason": "..."}"""
        from langchain_core.messages import HumanMessage, SystemMessage

        ticker = trade_packet.get("ticker")
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")

        research_section = (
            f"\nResearch Report:\n{json.dumps(research_report, indent=2)}"
            if research_report
            else "\nResearch Report: [Not available — ResearchAgent not yet implemented]"
        )

        human_msg = f"""Trade Signal:
Ticker:     {ticker}
Price:      {price}c
Action:     {action}
Confidence: {confidence}

Quant Report (all numbers pre-computed in Python):
{json.dumps(quant_report, indent=2)}
{research_section}
"""
        try:
            response = self.llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            clean = re.sub(r"```json|```", "", response.content).strip()
            return json.loads(clean)
        except Exception as e:
            return {"status": "PASS", "reason": f"Orchestrator error: {str(e)}"}

    # ── Public: full pipeline ────────────────────────────────────────────────
    def analyze_signal(self, trade_packet: dict) -> dict:
        """
        Full pipeline:
          1. Quant agent — calibration gap (Python math) + ESPN + nba_api context
          2. Research Agent placeholder — will provide news/injury/line data
          3. Orchestrator — synthesize into Trade Proposal (READY/PASS)
          4. Critic agent — adversarial portfolio-aware APPROVE/VETO (only on READY)

        Final status values:
          APPROVED  — passed quant + orchestrator + critic
          PASS      — rejected by orchestrator (no edge / bad data)
          VETOED    — passed orchestrator but critic killed it

        Execution is the caller's responsibility — this method only returns
        the decision dict.  The websocket client logs APPROVED trades to
        SQLite; src.settle checks back for P&L once games resolve.
        """
        ticker = trade_packet.get("ticker")
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")
        side   = "no" if action == "BET_NO" else "yes"

        # ── Step 1: Quant calibration gap analysis ────────────────────────────
        quant_report = self.quant.analyze(trade_packet)

        edge        = quant_report.get("calibration_gap")
        sample_size = quant_report.get("sample_size", 0)
        confidence  = _confidence(edge or 0, sample_size)
        kelly       = _kelly(edge or 0)

        # ── Step 1.5: Research Agent placeholder ─────────────────────────────
        # TODO: instantiate ResearchAgent (src/agents/researcher.py) and call it here.
        # It will pull news, injury reports, and betting line movements for the ticker.
        # Pass the result as research_report= into _orchestrate() below.
        research_report = None   # placeholder — ResearchAgent not yet implemented

        # ── Step 2: Orchestrator preliminary decision ─────────────────────────
        ruling = self._orchestrate(
            quant_report, confidence, sample_size, edge, trade_packet,
            research_report=research_report,
        )
        preliminary_status = ruling.get("status", "PASS")

        # ── Base decision object ──────────────────────────────────────────────
        decision = {
            "action":         action if preliminary_status == "READY" else "PASS",
            "ticker":         ticker,
            "side":           side if preliminary_status == "READY" else None,
            "confidence":     confidence,
            "edge":           edge,          # calibration_gap only — never raw yes_price
            "price":          price,
            "kelly_fraction": kelly if preliminary_status == "READY" else 0.0,
            "quant_summary":  quant_report,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "status":         preliminary_status,
            "reason":         ruling.get("reason", ""),
        }

        # ── Step 3: Critic review (only on READY) ─────────────────────────────
        if preliminary_status == "READY":
            decision = self.critic.review(trade_packet, decision)

        return decision
