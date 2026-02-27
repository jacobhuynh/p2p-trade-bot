"""
src/agents/orchestrator.py

The Lead Analyst — runs Quant and Sentiment in parallel, aggregates their
outputs, then forwards positive-edge signals to the Critic for final judgment.

Pipeline:
  trade_packet
    → [Quant.analyze(trade_packet) || Sentiment.enrich(copy)] in parallel
    → orchestrator aggregates: merge sentiment_context into trade_packet
    → Python gate (PASS if no edge / insufficient data)
    → READY only: synthesize quant + sentiment into one report
    → critic (receives synthesized report, quant report, sentiment context)

Models: Quant, Sentiment, and synthesis use claude-haiku-4-5; only the Critic
uses claude-sonnet-4-6. Requires ANTHROPIC_API_KEY.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
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


def _kelly(actual_win_rate: float | None, price: int | None, action: str | None) -> float:
    """
    Standard Kelly criterion for binary prediction market bets.

    f* = p - q / b
      where p = true win probability (actual_win_rate from historical calibration)
            q = 1 - p
            b = net odds = payout_cents / entry_cents

    For BET_NO: entry = 100 - yes_price, payout = yes_price
    For BET_YES: entry = yes_price, payout = 100 - yes_price

    Returns 0.0 on any bad input. Capped at KELLY_FRACTION_CAP.
    """
    if actual_win_rate is None or price is None or action is None:
        return 0.0
    if actual_win_rate <= 0:
        return 0.0

    entry_cents  = (100 - price) if action == "BET_NO" else price
    payout_cents = price         if action == "BET_NO" else (100 - price)

    if entry_cents <= 0 or payout_cents <= 0:
        return 0.0

    b   = payout_cents / entry_cents
    p   = actual_win_rate
    q   = 1.0 - p
    raw = p - q / b
    return round(max(0.0, min(raw, KELLY_FRACTION_CAP)), 4)


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
        from src.agents.sentiment_agent import SentimentAgent

        self.role    = "Lead Trading Orchestrator"
        self.sentiment_agent = SentimentAgent()
        self.quant   = QuantAgent()
        self.critic  = CriticAgent()

    def _synthesize(
        self,
        quant_report: dict,
        sentiment_context: str | None,
        ticker: str,
    ) -> str | None:
        """
        Combine quant report and sentiment into one short narrative for the Critic.
        Uses claude-haiku-4-5. Returns None on failure (pipeline continues with fallback).
        """
        gap = quant_report.get("calibration_gap")
        verdict = quant_report.get("verdict")
        sample_size = quant_report.get("sample_size", 0)
        quant_summary = quant_report.get("summary") or "(no quant summary)"
        sentiment = (sentiment_context or "").strip() or "No live sentiment available."

        prompt = f"""Ticker: {ticker}

Quant: calibration_gap={gap}, verdict={verdict}, sample_size={sample_size}. Summary: {quant_summary}

Sentiment (live ESPN/news): {sentiment}

Write one short paragraph (2-4 sentences) that combines the quantitative edge and verdict with the live sentiment. One coherent narrative for a trading Critic to review. Output only the paragraph, no labels."""
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage
            llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0)
            response = llm.invoke([HumanMessage(content=prompt)])
            out = (response.content or "").strip()
            return out if out else None
        except Exception:
            return f"Quant: {quant_summary} Sentiment: {sentiment}"

    # ── Public: full pipeline ────────────────────────────────────────────────
    def analyze_signal(self, trade_packet: dict) -> dict:
        """
        Full pipeline:
          0. Quant and Sentiment run in parallel (Quant on trade_packet, Sentiment on a copy).
          1. Orchestrator aggregates: merge sentiment_context from copy into trade_packet.
          2. Python-only gate — PASS only if no positive edge or insufficient data.
          3. READY only: synthesize quant + sentiment into one report.
          4. Critic agent — adversarial APPROVE/VETO; receives synthesized report, quant report, sentiment.

        Final status values:
          APPROVED  — passed quant gate + critic approved
          PASS      — Python gate: no positive edge or insufficient sample size
          VETOED    — passed Python gate but critic blocked it

        Execution is the caller's responsibility — this method only returns
        the decision dict.  The websocket client logs APPROVED trades to
        SQLite; src.settle checks back for P&L once games resolve.
        """
        # ── Step 0: Run Quant and Sentiment in parallel; then aggregate ───────
        packet_copy = dict(trade_packet)
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_quant = executor.submit(self.quant.analyze, trade_packet)
            future_sentiment = executor.submit(self.sentiment_agent.enrich, packet_copy)
            quant_report = future_quant.result()
            future_sentiment.result()
        trade_packet["sentiment_context"] = packet_copy.get("sentiment_context")
        sentiment_context = trade_packet.get("sentiment_context")

        ticker = trade_packet.get("ticker")
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")
        side   = "no" if action == "BET_NO" else "yes"

        edge             = quant_report.get("calibration_gap")
        actual_win_rate  = quant_report.get("actual_win_rate")
        sample_size      = quant_report.get("sample_size", 0)
        confidence       = _confidence(edge or 0, sample_size)
        kelly            = _kelly(actual_win_rate, price, action)

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
                "action":            "PASS",
                "ticker":           ticker,
                "side":             None,
                "confidence":       confidence,
                "edge":             edge,
                "price":            price,
                "kelly_fraction":    0.0,
                "quant_summary":     quant_report,
                "sentiment_context": sentiment_context,
                "timestamp":         datetime.now(timezone.utc).isoformat(),
                "status":           "PASS",
                "reason":           reason,
            }

        # ── Step 3: Synthesize quant + sentiment into one report for the Critic ─
        synthesized_report = self._synthesize(quant_report, sentiment_context, ticker)

        # ── Step 4: Build Trade Proposal — forward ALL positive-edge signals ──
        ready_reason = (
            "Positive calibration gap; sentiment context available — forwarding to Critic for final review"
            if sentiment_context else
            "Positive calibration gap — forwarding to Critic for final review"
        )
        trade_proposal = {
            "action":              action,
            "ticker":              ticker,
            "side":                side,
            "confidence":          confidence,
            "edge":                edge,
            "price":               price,
            "kelly_fraction":      kelly,
            "synthesized_report":  synthesized_report,
            "quant_summary":       quant_report,
            "sentiment_context":   sentiment_context,
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "status":              "READY",
            "reason":              ready_reason,
        }
        return self.critic.review(trade_packet, trade_proposal)
