"""
src/agents/rule_agents.py

Deterministic, rule-based implementations of QuantAgent, CriticAgent, and the
orchestrator decision logic.  Zero external API calls.

Used when LLM_MODE=rule (the default).  Each class exposes the same public
interface as the corresponding LLM agent so LeadAnalyst can swap them in without
touching analyze_signal().

Thresholds mirror orchestrator.py — any change there must be reflected here.
"""

from src.tools.duckdb_tool import (
    get_price_bucket_edge,
    get_longshot_bias_stats,
    get_historical_win_rate,
    get_market_volume_stats,
)

# ── Thresholds (must stay in sync with orchestrator.py) ─────────────────────
MIN_EDGE_HIGH      = 0.02
MIN_EDGE_MEDIUM    = 0.008
MIN_SAMPLE_SIZE    = 200
KELLY_FRACTION_CAP = 0.15


# ─────────────────────────────────────────────
# Rule-based QuantAgent
# ─────────────────────────────────────────────

class RuleQuantAgent:
    """
    Deterministic quant analysis using DuckDB price-bucket queries.
    Returns the same JSON schema as the LLM QuantAgent.
    """

    def analyze(self, trade_packet: dict) -> dict:
        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")

        # ── Implied probability for our side ────────────────────────────
        if action == "BET_NO":
            implied_prob = round((100 - price) / 100, 4)
        else:
            implied_prob = round(price / 100, 4)

        # ── Query price-bucket data ──────────────────────────────────────
        edge_data    = get_price_bucket_edge(price, action)
        bias_data    = get_longshot_bias_stats(price)
        win_data     = get_historical_win_rate(price)
        inverse_edge = get_price_bucket_edge(
            100 - price,
            "BET_YES" if action == "BET_NO" else "BET_NO",
        )

        actual_win_rate = edge_data.get("actual_win_rate")
        sample_size     = edge_data.get("sample_size", 0)
        no_win_rate     = bias_data.get("no_win_rate")

        # ── YES/NO asymmetry ─────────────────────────────────────────────
        yes_win = win_data.get("win_rate") if win_data else None
        no_wr   = inverse_edge.get("actual_win_rate")
        if yes_win is not None and no_wr is not None:
            yes_no_asymmetry = round(abs(yes_win - no_wr), 4)
        else:
            yes_no_asymmetry = None

        # ── Calibration gap ─────────────────────────────────────────────
        if actual_win_rate is not None:
            calibration_gap = round(actual_win_rate - implied_prob, 4)
            historical_edge = calibration_gap
        else:
            calibration_gap = None
            historical_edge = None

        # ── Verdict ─────────────────────────────────────────────────────
        if sample_size < 100 or actual_win_rate is None:
            verdict      = "INSUFFICIENT_DATA"
            data_quality = "INSUFFICIENT"
        elif calibration_gap is not None and calibration_gap > MIN_EDGE_HIGH and sample_size >= 200:
            verdict      = "EDGE_CONFIRMED"
            data_quality = "SUFFICIENT"
        elif calibration_gap is not None and calibration_gap > MIN_EDGE_MEDIUM and sample_size >= 100:
            verdict      = "EDGE_WEAK"
            data_quality = "SUFFICIENT"
        else:
            verdict      = "NO_EDGE"
            data_quality = "SUFFICIENT"

        # ── Summary ─────────────────────────────────────────────────────
        if verdict == "INSUFFICIENT_DATA":
            summary = f"Insufficient data at {price}c (n={sample_size})."
        elif verdict in ("EDGE_CONFIRMED", "EDGE_WEAK"):
            label = "Strong" if verdict == "EDGE_CONFIRMED" else "Weak"
            summary = (
                f"{label} edge at {price}c: gap={calibration_gap:.1%}, n={sample_size}."
            )
        else:
            summary = f"No tradeable edge at {price}c (gap={calibration_gap}, n={sample_size})."

        return {
            "historical_edge":  historical_edge,
            "actual_win_rate":  actual_win_rate,
            "implied_prob":     implied_prob,
            "no_win_rate":      no_win_rate,
            "yes_no_asymmetry": yes_no_asymmetry,
            "sample_size":      sample_size,
            "data_quality":     data_quality,
            "verdict":          verdict,
            "calibration_gap":  calibration_gap,
            "summary":          summary,
        }


# ─────────────────────────────────────────────
# Rule-based orchestrator decision function
# ─────────────────────────────────────────────

def rule_orchestrate(
    quant_report:    dict,
    confidence:      str,
    sample_size:     int,
    calibration_gap,
) -> dict:
    """
    Deterministic READY/PASS ruling.
    Returns {"status": "READY"/"PASS", "reason": "<str>"} — same shape as LLM ruling.
    """
    verdict = quant_report.get("verdict", "")

    if sample_size < MIN_SAMPLE_SIZE:
        return {
            "status": "PASS",
            "reason": f"Sample size {sample_size} < {MIN_SAMPLE_SIZE} minimum.",
        }

    if calibration_gap is None or calibration_gap <= 0:
        return {
            "status": "PASS",
            "reason": f"Calibration gap {calibration_gap} is None or non-positive.",
        }

    if verdict in {"EDGE_CONFIRMED", "EDGE_WEAK"} and confidence in {"HIGH", "MEDIUM"}:
        return {
            "status": "READY",
            "reason": (
                f"Verdict={verdict}, confidence={confidence}, "
                f"gap={calibration_gap:.4f}, n={sample_size}."
            ),
        }

    return {
        "status": "PASS",
        "reason": f"Verdict={verdict} / confidence={confidence} — criteria not met.",
    }


# ─────────────────────────────────────────────
# Rule-based CriticAgent
# ─────────────────────────────────────────────

class RuleCriticAgent:
    """
    Deterministic critic: checks hard veto rules, no LLM calls.
    Exposes the same .review(trade_packet, orchestrator_decision) interface.
    """

    def review(self, trade_packet: dict, orchestrator_decision: dict) -> dict:
        ticker      = trade_packet.get("ticker", "")
        quant       = orchestrator_decision.get("quant_summary", {})
        kelly       = orchestrator_decision.get("kelly_fraction", 0)
        actual_wr   = quant.get("actual_win_rate")
        sample_size = quant.get("sample_size", 0)
        cal_gap     = quant.get("calibration_gap")

        veto_reason = None
        concerns    = []

        # ── Hard veto rules (evaluated in priority order) ────────────────
        if actual_wr in (0.0, 1.0) and sample_size >= 200:
            veto_reason = (
                f"actual_win_rate of {actual_wr} across {sample_size} samples "
                "indicates data contamination, not genuine edge."
            )

        elif sample_size < 100:
            veto_reason = (
                f"Sample size {sample_size} is below minimum threshold of 100."
            )

        elif kelly > KELLY_FRACTION_CAP:
            veto_reason = (
                f"kelly_fraction {kelly:.4f} exceeds cap of {KELLY_FRACTION_CAP}."
            )

        else:
            # ── Liquidity check (skipped when DB has no data for ticker) ──
            vol_stats     = get_market_volume_stats(ticker)
            volume        = vol_stats.get("volume", 0)
            open_interest = vol_stats.get("open_interest", 0)
            # Only apply the liquidity veto when we actually have volume data.
            # When both are 0 it means the ticker isn't in the DB (test / offline).
            has_vol_data = (volume > 0) or (open_interest > 0)

            if has_vol_data:
                if volume == 0:
                    veto_reason = (
                        "Market volume is 0 — cannot get filled at this price."
                    )
                elif open_interest < 500:
                    veto_reason = (
                        f"Open interest of {open_interest} is below 500 — "
                        "market is illiquid."
                    )

            if not veto_reason:
                concerns.append("Rule-based critic: no structural issues detected.")

        # ── Build critique dict ──────────────────────────────────────────
        if veto_reason:
            critique = {
                "decision":    "VETO",
                "veto_reason": veto_reason,
                "concerns":    [veto_reason],
                "risk_score":  9,
                "summary":     f"VETO: {veto_reason}",
            }
        else:
            risk_score = 3 if (cal_gap or 0) > MIN_EDGE_HIGH else 5
            critique = {
                "decision":    "APPROVE",
                "veto_reason": None,
                "concerns":    concerns,
                "risk_score":  risk_score,
                "summary":     "Rule-based critic approved: no veto conditions triggered.",
            }

        final_status = "APPROVED" if critique["decision"] == "APPROVE" else "VETOED"

        return {
            **orchestrator_decision,
            "status": final_status,
            "action": orchestrator_decision["action"] if final_status == "APPROVED" else "PASS",
            "critic": {
                "decision":    critique["decision"],
                "veto_reason": critique.get("veto_reason"),
                "concerns":    critique.get("concerns", []),
                "risk_score":  critique.get("risk_score"),
                "summary":     critique.get("summary"),
            },
        }
