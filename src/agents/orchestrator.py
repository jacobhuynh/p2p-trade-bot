"""
src/agents/orchestrator.py

The Lead Analyst — orchestrates the quant agent, then forwards every
positive-edge signal directly to the Critic for final judgment.

Pipeline:
  trade_packet
    → quant  (calibration gap + ESPN context + nba_api stats)
    → [Research Agent placeholder — not yet implemented]
    → orchestrator (pure data synthesizer: compute Kelly/Confidence, build Trade Proposal)
    → critic (adversarial APPROVE/VETO — receives all positive-edge signals)

Always uses Claude (claude-sonnet-4-6). Requires ANTHROPIC_API_KEY.
"""

import json
import re
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# THRESHOLDS
# Calibrated to real prediction market edge sizes.
# Real edges are small — 1.5% is strong, 0.75% is weak but real.
# ─────────────────────────────────────────────

MIN_EDGE_HIGH      = 0.015   # 1.5%+ calibration gap = strong edge
MIN_EDGE_MEDIUM    = 0.0075  # 0.75%+ calibration gap = weak but real
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


class LeadAnalyst:
    def __init__(self):
        from src.agents.quant import QuantAgent
        from src.agents.critic import CriticAgent

        self.role   = "Lead Trading Orchestrator"
        self.quant  = QuantAgent()
        self.critic = CriticAgent()

    # ── Public: full pipeline ────────────────────────────────────────────────
    def analyze_signal(self, trade_packet: dict) -> dict:
        """
        Full pipeline:
          1. Quant agent — calibration gap (Python math) + ESPN + nba_api context
          2. Python-only gate — PASS only if no positive edge or insufficient data
          3. Critic agent — adversarial portfolio-aware APPROVE/VETO on all valid signals

        Final status values:
          APPROVED  — passed quant gate + critic approved
          PASS      — Python gate: no positive edge or insufficient sample size
          VETOED    — passed Python gate but critic blocked it

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

        # ── Step 2: Python-only gate — no LLM decision ────────────────────────
        # Only PASS if there is literally no positive edge or insufficient data.
        # Everything else goes directly to the Critic for final judgment.
        if not edge or edge <= 0:
            reason    = "No positive calibration gap — no edge to trade"
            gate_pass = True
        elif sample_size < MIN_SAMPLE_SIZE:
            reason    = f"Insufficient sample size ({sample_size} < {MIN_SAMPLE_SIZE})"
            gate_pass = True
        else:
            gate_pass = False

        if gate_pass:
            return {
                "action":         "PASS",
                "ticker":         ticker,
                "side":           None,
                "confidence":     confidence,
                "edge":           edge,
                "price":          price,
                "kelly_fraction": 0.0,
                "quant_summary":  quant_report,
                "timestamp":      datetime.now(timezone.utc).isoformat(),
                "status":         "PASS",
                "reason":         reason,
            }

        # ── Step 3: Build Trade Proposal — forward ALL positive-edge signals ──
        trade_proposal = {
            "action":         action,
            "ticker":         ticker,
            "side":           side,
            "confidence":     confidence,
            "edge":           edge,
            "price":          price,
            "kelly_fraction": kelly,
            "quant_summary":  quant_report,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "status":         "READY",
            "reason":         "Positive calibration gap — forwarding to Critic for final review",
        }
        return self.critic.review(trade_packet, trade_proposal)
