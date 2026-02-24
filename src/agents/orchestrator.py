"""
src/agents/orchestrator.py

The Lead Analyst — orchestrates the quant agent, makes a preliminary
READY/PASS decision, then sends READY decisions to the critic for
final approval before returning an execution-ready decision object.

Pipeline:
  trade_packet → quant (calibration gap) → orchestrator (READY/PASS) → critic (APPROVE/VETO)

Agent selection is controlled by LLM_MODE (src/config.py):
  LLM_MODE=rule       — deterministic rule-based agents (default, no API keys needed)
  LLM_MODE=anthropic  — ChatAnthropic-powered agents (requires ANTHROPIC_API_KEY)
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


# System prompt used only in LLM mode
_SYSTEM_PROMPT = """You are a lead trading analyst for a Kalshi prediction market trading bot.
You receive a trade signal and a quantitative calibration gap analysis.
Make the preliminary READY or PASS decision.

IMPORTANT CONTEXT: Real prediction market edges are small. A 2% calibration gap is
considered strong. A 0.8% gap is weak but real. Do not expect large edges.

Respond ONLY with a JSON object — no extra text:
{{
    "status":  "READY" or "PASS",
    "reason":  "<one sentence justification>"
}}

Decision rules:
- READY if: verdict is EDGE_CONFIRMED or EDGE_WEAK, AND confidence is HIGH or MEDIUM
- PASS if: verdict is NO_EDGE or INSUFFICIENT_DATA, or confidence is LOW
- Always PASS if sample_size < {min_sample}
- Always PASS if calibration_gap is null or <= 0
""".format(min_sample=MIN_SAMPLE_SIZE)


class LeadAnalyst:
    def __init__(self):
        from src.config import LLM_MODE

        self.role     = "Lead Trading Orchestrator"
        self.llm_mode = LLM_MODE

        if LLM_MODE == "rule":
            # Zero external calls — import only local modules
            from src.agents.rule_agents import RuleQuantAgent, RuleCriticAgent
            self.quant  = RuleQuantAgent()
            self.critic = RuleCriticAgent()
            self.llm    = None
        else:
            # anthropic mode — lazy-import so missing package only fails here
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415
            from src.agents.quant import QuantAgent
            from src.agents.critic import CriticAgent
            self.quant  = QuantAgent()
            self.critic = CriticAgent()
            self.llm    = ChatAnthropic(
                model="claude-sonnet-4-6",
                temperature=0,
            )

    # ── Internal: orchestrator decision step ────────────────────────────────
    def _orchestrate(
        self,
        quant_report:    dict,
        confidence:      str,
        sample_size:     int,
        calibration_gap,
        trade_packet:    dict,
    ) -> dict:
        """Returns {"status": "READY"/"PASS", "reason": "..."}"""

        if self.llm is None:
            # Rule mode — pure Python logic, no network
            from src.agents.rule_agents import rule_orchestrate
            return rule_orchestrate(quant_report, confidence, sample_size, calibration_gap)

        # ── LLM (anthropic) mode ─────────────────────────────────────────
        from langchain_core.messages import HumanMessage, SystemMessage

        ticker = trade_packet.get("ticker")
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")

        human_msg = f"""Trade Signal:
Ticker:     {ticker}
Price:      {price}c
Action:     {action}
Confidence: {confidence}

Quant Report:
{json.dumps(quant_report, indent=2)}
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
          1. Quant agent — calibration gap analysis by price bucket
          2. Orchestrator — preliminary READY/PASS (rule or LLM)
          3. Critic agent — adversarial APPROVE/VETO (only on READY)

        Final status values:
          APPROVED  — passed quant + orchestrator + critic → send to trade_manager
          PASS      — rejected by orchestrator (no edge / bad data)
          VETOED    — passed orchestrator but critic killed it
        """
        ticker = trade_packet.get("ticker")
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")
        side   = "no" if action == "BET_NO" else "yes"

        # ── Step 1: Quant calibration gap analysis ──────────────────────
        quant_report = self.quant.analyze(trade_packet)

        edge        = quant_report.get("calibration_gap")
        sample_size = quant_report.get("sample_size", 0)
        confidence  = _confidence(edge or 0, sample_size)
        kelly       = _kelly(edge or 0)

        # ── Step 2: Orchestrator preliminary decision ───────────────────
        ruling = self._orchestrate(
            quant_report, confidence, sample_size, edge, trade_packet
        )
        preliminary_status = ruling.get("status", "PASS")

        # ── Base decision object ────────────────────────────────────────
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

        # ── Step 3: Critic review (only on READY) ──────────────────────
        if preliminary_status == "READY":
            decision = self.critic.review(trade_packet, decision)

        # ── Step 4: Execution (only on APPROVED) ────────────────────────
        if decision.get("status") == "APPROVED" and decision.get("action") in ("BET_YES", "BET_NO"):
            from src.config import EXECUTION_MODE
            if EXECUTION_MODE == "paper":
                from src.execution.trade_manager import PaperTradeManager
                report = PaperTradeManager().execute(decision, trade_packet)
                decision["execution"] = report
            elif EXECUTION_MODE == "live":
                from src.execution.trade_manager import LiveTradeManager
                LiveTradeManager().execute(decision, trade_packet)

        return decision
