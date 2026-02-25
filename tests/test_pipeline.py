"""
tests/test_pipeline.py

Tests the full pipeline: bouncer -> quant -> orchestrator -> critic

Run modes:
    pytest tests/test_pipeline.py -v -s          # unit tests (LLM calls mocked — no API key needed)
    python tests/test_pipeline.py                # same as above
    python tests/test_pipeline.py --live         # real DB + real Claude (requires ANTHROPIC_API_KEY)
"""

import csv
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Patch targets — always the real LLM agent classes
QUANT_ANALYZE_PATCH = "src.agents.quant.QuantAgent.analyze"
CRITIC_REVIEW_PATCH = "src.agents.critic.CriticAgent.review"

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
    label  = {"APPROVED": "OK", "VETOED": "VETO", "PASS": "PASS"}.get(status, "?")
    print(f"\n{label} Status:     {status}")
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
        print("OK NBA longshot YES (14c) -> BET_NO")

        result = process_trade(VALID_NBA_LONGSHOT_NO)
        assert result is not None and result["action"] == "BET_YES"
        print("OK NBA longshot NO (86c) -> BET_YES")

        result = process_trade(INVALID_NBA_MIDDLE_PRICE)
        assert result is None
        print("OK NBA middle price (55c) -> rejected")

        result = process_trade(INVALID_NON_NBA)
        assert result is None
        print("OK Non-NBA ticker -> rejected")

        result = process_trade(INVALID_EMPTY)
        assert result is None
        print("OK Empty payload -> rejected")

    print("\nOK All bouncer tests passed.")


# ─────────────────────────────────────────────
# QUANT TESTS
# ─────────────────────────────────────────────

def test_quant_price_bucket_query():
    from src.tools.duckdb_tool import get_price_bucket_edge, get_longshot_bias_stats

    print_section("QUANT PRICE BUCKET TESTS")

    edge = get_price_bucket_edge(14, "BET_NO")
    assert isinstance(edge, dict), "Should always return a dict"
    assert "error" in edge or "edge" in edge, "Must have 'edge' key or 'error' key"
    if edge.get("sample_size", 0) > 0:
        assert isinstance(edge["actual_win_rate"], float), "actual_win_rate should be float"
        assert isinstance(edge["edge"], float), "edge should be float"
        assert isinstance(edge["implied_prob"], float), "implied_prob should be float"
        assert isinstance(edge["sample_size"], int), "sample_size should be int"
    print(f"OK Price bucket edge at 14c BET_NO: {edge}")

    bias = get_longshot_bias_stats(14)
    assert isinstance(bias, dict), "Should always return a dict"
    assert "error" in bias or "no_win_rate" in bias, "Must have 'no_win_rate' key or 'error' key"
    if bias.get("sample_size", 0) > 0:
        assert isinstance(bias["no_win_rate"], float), "no_win_rate should be float"
        assert isinstance(bias["avg_price"], float), "avg_price should be float"
        assert isinstance(bias["sample_size"], int), "sample_size should be int"
    print(f"OK Longshot bias stats at <=14c: {bias}")

    print("\nOK All quant price bucket tests passed.")


# ─────────────────────────────────────────────
# FULL PIPELINE TESTS
# ─────────────────────────────────────────────

def test_pipeline_approved():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — APPROVED")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH,  return_value=MOCK_QUANT_EDGE_CONFIRMED), \
         patch(CRITIC_REVIEW_PATCH,  side_effect=_mock_critic_approve):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]             == "APPROVED"
        assert decision["action"]             == "BET_NO"
        assert decision["confidence"]         in ("HIGH", "MEDIUM")
        assert decision["kelly_fraction"]     >  0
        assert decision["critic"]["decision"] == "APPROVE"
        # Execution is now handled by the caller (websocket_client + TradeLogger),
        # not embedded in the orchestrator — no "execution" key expected here.

    print("\nOK Approved pipeline test passed.")


def test_pipeline_vetoed_contamination():
    """
    Contamination path: quant returns win_rate=1.0.
    LLM orchestrator is mocked to return READY, then LLM critic is mocked to VETO.
    """
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst
    from langchain_core.messages import AIMessage

    print_section("PIPELINE TEST — VETOED (data contamination)")

    mock_llm_resp = AIMessage(
        content='{"status": "READY", "reason": "EDGE_CONFIRMED verdict with sufficient sample size"}'
    )

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_CONTAMINATED), \
         patch("langchain_anthropic.ChatAnthropic.invoke", return_value=mock_llm_resp), \
         patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_veto_contamination):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]             == "VETOED"
        assert decision["action"]             == "PASS"
        assert decision["critic"]["decision"] == "VETO"

    print("\nOK Contamination veto test passed.")


def test_pipeline_pass_no_edge():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — PASS (no edge)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_NO_EDGE), \
         patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_approve) as mock_critic:

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"] == "PASS"
        assert decision["action"] == "PASS"
        mock_critic.assert_not_called()

    print("\nOK No edge pass test passed.")


def test_pipeline_pass_insufficient_data():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — PASS (insufficient data)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_INSUFFICIENT), \
         patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_approve) as mock_critic:

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"] == "PASS"
        mock_critic.assert_not_called()

    print("\nOK Insufficient data pass test passed.")


def test_pipeline_vetoed_liquidity():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — VETOED (illiquid market)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_EDGE_CONFIRMED), \
         patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_veto_liquidity):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]             == "VETOED"
        assert decision["critic"]["decision"] == "VETO"
        assert (
            "volume"  in decision["critic"]["veto_reason"].lower() or
            "liquid"  in decision["critic"]["veto_reason"].lower()
        )

    print("\nOK Liquidity veto test passed.")


def test_pipeline_weak_edge():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — WEAK EDGE")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_EDGE_WEAK), \
         patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_approve):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        # EDGE_WEAK with sample_size=75 < MIN_SAMPLE_SIZE=200 → LOW confidence → PASS
        assert decision["status"]     == "PASS"
        assert decision["confidence"] == "LOW"

    print("\nOK Weak edge test passed.")


# ─────────────────────────────────────────────
# PAPER TRADING E2E TEST
# ─────────────────────────────────────────────

def test_paper_trading_e2e():
    """
    End-to-end trade logger test.

    Exercises the full pipeline → TradeLogger path:
      - Pipeline returns APPROVED decision
      - TradeLogger.log_trade() persists it to a temp SQLite DB as PENDING_RESOLUTION
      - TradeLogger.evaluate_trade() marks it EVALUATED with correct P&L
      - summary() returns the right aggregate stats
    """
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst
    from src.execution.trade_logger import TradeLogger

    print_section("TRADE LOGGER E2E TEST")

    tmp_db = tempfile.mktemp(prefix="p2p_trade_logger_test_", suffix=".db")
    try:
        with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
             patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_EDGE_CONFIRMED), \
             patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_approve):

            trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
            decision     = LeadAnalyst().analyze_signal(trade_packet)
            print_decision(decision)

        # ── Verify decision ──────────────────────────────────────────────
        assert decision["status"] == "APPROVED"
        assert decision["action"] == "BET_NO"

        # ── Log the trade ────────────────────────────────────────────────
        logger   = TradeLogger(db_path=tmp_db)
        trade_id = logger.log_trade(decision, trade_packet, stake=10.0)
        assert trade_id >= 1, "log_trade should return a positive row id"

        open_trades = logger.open_trades()
        assert len(open_trades) == 1
        row = open_trades[0]
        assert row["status"]  == "PENDING_RESOLUTION"
        assert row["action"]  == "BET_NO"
        assert row["side"]    == "no"
        assert row["cost_usd"] > 0

        print(f"   Logged trade #{trade_id}: {row['ticker']}  cost=${row['cost_usd']:.2f}")

        # ── Evaluate the trade (NO wins — correct call for BET_NO on a 14c YES longshot) ──
        settled = logger.evaluate_trade(trade_id, result="no")
        assert settled["won"]      is True
        assert settled["pnl_usd"]  > 0   # payout=$contracts, cost<$contracts → profit
        assert settled["payout_usd"] == row["contracts"] * 1.0

        open_after = logger.open_trades()
        assert len(open_after) == 0, "No open trades should remain after settlement"

        # ── Summary ──────────────────────────────────────────────────────
        s = logger.summary()
        assert s["n_trades"] == 1
        assert s["n_wins"]   == 1
        assert s["win_rate"] == 1.0
        assert s["total_pnl"] > 0

        print(f"   Settled: won={settled['won']}  pnl=${settled['pnl_usd']:.2f}")
        print(f"   Summary: {s}")

    finally:
        Path(tmp_db).unlink(missing_ok=True)

    print("\nOK Trade logger e2e test passed.")


# ─────────────────────────────────────────────
# LIVE TEST
# ─────────────────────────────────────────────

def test_pipeline_live():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("LIVE PIPELINE TEST (real DB + real Claude agents)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET):
        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        assert trade_packet is not None

        print(f"[Trade Packet]\n{json.dumps(trade_packet, indent=2)}")

        decision = LeadAnalyst().analyze_signal(trade_packet)
        print(f"\n[Decision]\n{json.dumps(decision, indent=2)}")
        print_decision(decision)

        assert decision["status"] in ("APPROVED", "VETOED", "PASS")

    print("\nOK Live pipeline test complete.")


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
    test_paper_trading_e2e()

    if "--live" in sys.argv:
        test_pipeline_live()
    else:
        print("\n[Tip] Run with --live to also exercise real DB queries and real Claude agents")

    print(f"\n{'='*60}")
    print(f"  All tests passed")
    print(f"{'='*60}")
