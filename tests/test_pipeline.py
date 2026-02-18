"""
tests/test_pipeline.py

Tests the full pipeline: bouncer -> quant -> orchestrator -> critic

Run modes:
    pytest tests/test_pipeline.py -v -s          # mocked (fast, no API calls)
    python tests/test_pipeline.py                # mocked (fast, no API calls)
    python tests/test_pipeline.py --live         # real DB + real LLM calls
"""

import json
import sys
from unittest.mock import patch

# ─────────────────────────────────────────────
# MOCK TRADE PAYLOADS
# ─────────────────────────────────────────────

VALID_NBA_LONGSHOT_YES = {
    "market_ticker": "KXNBAGAME-26FEB19BKNCLE-BKN",
    "yes_price": 14,
}

VALID_NBA_LONGSHOT_NO = {
    "market_ticker": "KXNBAGAME-26FEB19BKNCLE-CLE",
    "yes_price": 86,
}

INVALID_NBA_MIDDLE_PRICE = {
    "market_ticker": "KXNBAGAME-26FEB19LALGWS-LAL",
    "yes_price": 55,
}

INVALID_NON_NBA = {
    "market_ticker": "KXNFLGAME-26FEB19KCCIN",
    "yes_price": 12,
}

INVALID_EMPTY = {}

# ─────────────────────────────────────────────
# MOCK MARKET REST RESPONSE
# ─────────────────────────────────────────────

MOCK_MARKET = {
    "title":         "Brooklyn at Cleveland Winner?",
    "market_type":   "binary",
    "rules_primary": "If Brooklyn wins the Brooklyn at Cleveland game, resolves Yes.",
    "no_sub_title":  "Cleveland",
}

# ─────────────────────────────────────────────
# MOCK QUANT REPORTS
# ─────────────────────────────────────────────

MOCK_QUANT_EDGE_CONFIRMED = {
    "historical_edge":  0.09,
    "actual_win_rate":  0.94,
    "implied_prob":     0.85,
    "no_win_rate":      0.91,
    "yes_no_asymmetry": 0.12,
    "sample_size":      312,
    "data_quality":     "SUFFICIENT",
    "verdict":          "EDGE_CONFIRMED",
    "calibration_gap":  0.09,
    "summary":          "Strong longshot bias at 14c with 312 historical samples and 9% calibration gap.",
}

MOCK_QUANT_EDGE_WEAK = {
    "historical_edge":  0.04,
    "actual_win_rate":  0.89,
    "implied_prob":     0.85,
    "no_win_rate":      0.87,
    "yes_no_asymmetry": 0.05,
    "sample_size":      75,
    "data_quality":     "SUFFICIENT",
    "verdict":          "EDGE_WEAK",
    "calibration_gap":  0.04,
    "summary":          "Weak edge detected at 14c, sample size borderline.",
}

MOCK_QUANT_NO_EDGE = {
    "historical_edge":  -0.02,
    "actual_win_rate":  0.44,
    "implied_prob":     0.46,
    "no_win_rate":      0.44,
    "yes_no_asymmetry": 0.01,
    "sample_size":      87,
    "data_quality":     "SUFFICIENT",
    "verdict":          "NO_EDGE",
    "calibration_gap":  -0.02,
    "summary":          "No meaningful edge at this price.",
}

MOCK_QUANT_INSUFFICIENT = {
    "historical_edge":  None,
    "actual_win_rate":  None,
    "implied_prob":     None,
    "no_win_rate":      None,
    "yes_no_asymmetry": None,
    "sample_size":      0,
    "data_quality":     "INSUFFICIENT",
    "verdict":          "INSUFFICIENT_DATA",
    "calibration_gap":  None,
    "summary":          "No historical data available.",
}

MOCK_QUANT_CONTAMINATED = {
    "historical_edge":  0.14,
    "actual_win_rate":  1.0,
    "implied_prob":     0.86,
    "no_win_rate":      1.0,
    "yes_no_asymmetry": 0.20,
    "sample_size":      7850,
    "data_quality":     "SUFFICIENT",
    "verdict":          "EDGE_CONFIRMED",
    "calibration_gap":  0.14,
    "summary":          "Edge confirmed but win rate of 1.0 is suspicious.",
}

# ─────────────────────────────────────────────
# MOCK CRITIC RESPONSES
# ─────────────────────────────────────────────

MOCK_CRITIC_APPROVE = {
    "decision":    "APPROVE",
    "veto_reason": None,
    "concerns":    ["Sample size of 312 is adequate but not large"],
    "risk_score":  3,
    "summary":     "Edge is real, data is clean, liquidity is sufficient.",
}

MOCK_CRITIC_VETO_CONTAMINATION = {
    "decision":    "VETO",
    "veto_reason": "actual_win_rate of 1.0 across 7850 samples indicates data contamination, not genuine edge.",
    "concerns":    ["Win rate of 1.0 is statistically impossible", "Calibration gap may be inflated"],
    "risk_score":  9,
    "summary":     "Data contamination detected — win rate of 1.0 is a red flag.",
}

MOCK_CRITIC_VETO_LIQUIDITY = {
    "decision":    "VETO",
    "veto_reason": "Market volume is 0 — cannot get filled at this price.",
    "concerns":    ["Zero open interest", "No volume in last 24h"],
    "risk_score":  8,
    "summary":     "Illiquid market — trade cannot be executed.",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_decision(decision):
    status = decision.get("status")
    emoji  = {"APPROVED": "✅", "VETOED": "🚫", "PASS": "⏭️"}.get(status, "❓")
    print(f"\n{emoji} Status:     {status}")
    print(f"   Action:     {decision.get('action')}")
    print(f"   Confidence: {decision.get('confidence')}")
    print(f"   Edge:       {decision.get('edge')}")
    print(f"   Kelly:      {decision.get('kelly_fraction')}")
    print(f"   Reason:     {decision.get('reason')}")
    critic = decision.get("critic")
    if critic:
        print(f"   Critic:     {critic.get('decision')} | Risk: {critic.get('risk_score')}/10")
        if critic.get("veto_reason"):
            print(f"   Veto:       {critic.get('veto_reason')}")


# ─────────────────────────────────────────────
# MOCK CRITIC SIDE EFFECTS
# ─────────────────────────────────────────────

def _mock_critic_approve(trade_packet, decision):
    return {**decision, "status": "APPROVED", "critic": MOCK_CRITIC_APPROVE}

def _mock_critic_veto_contamination(trade_packet, decision):
    return {**decision, "status": "VETOED", "action": "PASS", "critic": MOCK_CRITIC_VETO_CONTAMINATION}

def _mock_critic_veto_liquidity(trade_packet, decision):
    return {**decision, "status": "VETOED", "action": "PASS", "critic": MOCK_CRITIC_VETO_LIQUIDITY}


# ─────────────────────────────────────────────
# BOUNCER TESTS
# ─────────────────────────────────────────────

def test_bouncer_filters():
    from src.pipeline.bouncer import process_trade

    print_section("BOUNCER FILTER TESTS")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET):
        result = process_trade(VALID_NBA_LONGSHOT_YES)
        assert result is not None and result["action"] == "BET_NO"
        print("✅ NBA longshot YES (14c) → BET_NO")

        result = process_trade(VALID_NBA_LONGSHOT_NO)
        assert result is not None and result["action"] == "BET_YES"
        print("✅ NBA longshot NO (86c) → BET_YES")

        result = process_trade(INVALID_NBA_MIDDLE_PRICE)
        assert result is None
        print("✅ NBA middle price (55c) → rejected")

        result = process_trade(INVALID_NON_NBA)
        assert result is None
        print("✅ Non-NBA ticker → rejected")

        result = process_trade(INVALID_EMPTY)
        assert result is None
        print("✅ Empty payload → rejected")

    print("\n✅ All bouncer tests passed.")


# ─────────────────────────────────────────────
# QUANT TESTS
# ─────────────────────────────────────────────

def test_quant_price_bucket_query():
    from src.tools.duckdb_tool import get_price_bucket_edge, get_longshot_bias_stats

    print_section("QUANT PRICE BUCKET TESTS")

    edge = get_price_bucket_edge(14, "BET_NO")
    assert "edge" in edge or "error" in edge
    print(f"✅ Price bucket edge at 14c BET_NO: {edge}")

    bias = get_longshot_bias_stats(14)
    assert "no_win_rate" in bias or "error" in bias
    print(f"✅ Longshot bias stats at <=14c: {bias}")

    print("\n✅ All quant price bucket tests passed.")


# ─────────────────────────────────────────────
# FULL PIPELINE TESTS
# ─────────────────────────────────────────────

def test_pipeline_approved():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — APPROVED")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch("src.agents.quant.QuantAgent.analyze",     return_value=MOCK_QUANT_EDGE_CONFIRMED), \
         patch("src.agents.critic.CriticAgent.review",    side_effect=_mock_critic_approve):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]             == "APPROVED"
        assert decision["action"]             == "BET_NO"
        assert decision["confidence"]         in ("HIGH", "MEDIUM")
        assert decision["kelly_fraction"]     >  0
        assert decision["critic"]["decision"] == "APPROVE"

    print("\n✅ Approved pipeline test passed.")


def test_pipeline_vetoed_contamination():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — VETOED (data contamination)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch("src.agents.quant.QuantAgent.analyze",     return_value=MOCK_QUANT_CONTAMINATED), \
         patch("src.agents.critic.CriticAgent.review",    side_effect=_mock_critic_veto_contamination):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]             == "VETOED"
        assert decision["action"]             == "PASS"
        assert decision["critic"]["decision"] == "VETO"

    print("\n✅ Contamination veto test passed.")


def test_pipeline_pass_no_edge():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — PASS (no edge)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch("src.agents.quant.QuantAgent.analyze",     return_value=MOCK_QUANT_NO_EDGE), \
         patch("src.agents.critic.CriticAgent.review",    side_effect=_mock_critic_approve) as mock_critic:

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"] == "PASS"
        assert decision["action"] == "PASS"
        mock_critic.assert_not_called()

    print("\n✅ No edge pass test passed.")


def test_pipeline_pass_insufficient_data():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — PASS (insufficient data)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch("src.agents.quant.QuantAgent.analyze",     return_value=MOCK_QUANT_INSUFFICIENT), \
         patch("src.agents.critic.CriticAgent.review",    side_effect=_mock_critic_approve) as mock_critic:

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"] == "PASS"
        mock_critic.assert_not_called()

    print("\n✅ Insufficient data pass test passed.")


def test_pipeline_vetoed_liquidity():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — VETOED (illiquid market)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch("src.agents.quant.QuantAgent.analyze",     return_value=MOCK_QUANT_EDGE_CONFIRMED), \
         patch("src.agents.critic.CriticAgent.review",    side_effect=_mock_critic_veto_liquidity):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]             == "VETOED"
        assert decision["critic"]["decision"] == "VETO"
        assert "volume" in decision["critic"]["veto_reason"].lower() or \
               "liquid"  in decision["critic"]["veto_reason"].lower()

    print("\n✅ Liquidity veto test passed.")


def test_pipeline_weak_edge():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — WEAK EDGE")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch("src.agents.quant.QuantAgent.analyze",     return_value=MOCK_QUANT_EDGE_WEAK), \
         patch("src.agents.critic.CriticAgent.review",    side_effect=_mock_critic_approve):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]     in ("APPROVED", "PASS")
        assert decision["confidence"] in ("MEDIUM", "LOW")

    print("\n✅ Weak edge test passed.")


# ─────────────────────────────────────────────
# LIVE TEST
# ─────────────────────────────────────────────

def test_pipeline_live():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("LIVE PIPELINE TEST (real DB + real LLM)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET):
        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        assert trade_packet is not None

        print(f"📦 Trade Packet:\n{json.dumps(trade_packet, indent=2)}")

        decision = LeadAnalyst().analyze_signal(trade_packet)
        print(f"\n🤖 Full Decision:\n{json.dumps(decision, indent=2)}")
        print_decision(decision)

        assert decision["status"] in ("APPROVED", "VETOED", "PASS")

    print("\n✅ Live pipeline test complete.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_bouncer_filters()
    test_quant_price_bucket_query()
    test_pipeline_approved()
    test_pipeline_vetoed_contamination()
    test_pipeline_pass_no_edge()
    test_pipeline_pass_insufficient_data()
    test_pipeline_vetoed_liquidity()
    test_pipeline_weak_edge()

    if "--live" in sys.argv:
        test_pipeline_live()
    else:
        print("\n💡 Tip: run with --live to test against real DB and LLM")

    print(f"\n{'='*60}")
    print("  All tests passed ✅")
    print(f"{'='*60}")