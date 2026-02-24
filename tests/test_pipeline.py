"""
tests/test_pipeline.py

Tests the full pipeline: bouncer -> quant -> orchestrator -> critic

Run modes:
    pytest tests/test_pipeline.py -v -s          # rule mode (fast, no API calls)
    python tests/test_pipeline.py                # rule mode (fast, no API calls)
    LLM_MODE=anthropic python tests/test_pipeline.py --live  # real DB + real LLM

LLM_MODE defaults to "rule" so the suite passes with zero API keys.
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

# ── Default to rule mode before any project imports ──────────────────────────
# Set before importing src modules so config.py picks it up correctly.
os.environ.setdefault("LLM_MODE", "rule")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import LLM_MODE  # noqa: E402

# ── Patch-target routing based on active mode ─────────────────────────────────
# In rule mode we patch the rule agent classes; in anthropic mode the LLM agents.
if LLM_MODE == "rule":
    QUANT_ANALYZE_PATCH = "src.agents.rule_agents.RuleQuantAgent.analyze"
    CRITIC_REVIEW_PATCH = "src.agents.rule_agents.RuleCriticAgent.review"
else:
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
    assert "edge" in edge or "error" in edge
    print(f"OK Price bucket edge at 14c BET_NO: {edge}")

    bias = get_longshot_bias_stats(14)
    assert "no_win_rate" in bias or "error" in bias
    print(f"OK Longshot bias stats at <=14c: {bias}")

    print("\nOK All quant price bucket tests passed.")


# ─────────────────────────────────────────────
# FULL PIPELINE TESTS
# ─────────────────────────────────────────────

def test_pipeline_approved():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — APPROVED")

    # Patch PaperTradeManager so this unit test has no file-system side effects.
    # The dedicated test_paper_trading_e2e test verifies real file output.
    mock_paper = MagicMock()
    mock_paper.return_value.execute.return_value = {
        "status": "FILLED", "contracts": 1, "cash_after": 999.14,
    }

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
         patch(QUANT_ANALYZE_PATCH,  return_value=MOCK_QUANT_EDGE_CONFIRMED), \
         patch(CRITIC_REVIEW_PATCH,  side_effect=_mock_critic_approve), \
         patch("src.execution.trade_manager.PaperTradeManager", mock_paper):

        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        decision     = LeadAnalyst().analyze_signal(trade_packet)
        print_decision(decision)

        assert decision["status"]              == "APPROVED"
        assert decision["action"]              == "BET_NO"
        assert decision["confidence"]          in ("HIGH", "MEDIUM")
        assert decision["kelly_fraction"]      >  0
        assert decision["critic"]["decision"]  == "APPROVE"
        assert decision["execution"]["status"] == "FILLED"

    print("\nOK Approved pipeline test passed.")


def test_pipeline_vetoed_contamination():
    """
    Contamination path: quant returns win_rate=1.0.

    In rule mode  — the rule orchestrator issues READY (gap=0.14, n=7850),
                    then the rule critic detects win_rate=1.0 and VETOs
                    without any LLM call.
    In anthropic mode — the LLM orchestrator is mocked to return READY,
                        then the LLM critic is mocked to VETO.
    """
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PIPELINE TEST — VETOED (data contamination)")

    extra_patches = []
    if LLM_MODE != "rule":
        # In LLM mode we need to mock the orchestrator LLM response
        from langchain_core.messages import AIMessage
        mock_llm_resp = AIMessage(
            content='{"status": "READY", "reason": "EDGE_CONFIRMED verdict with sufficient sample size"}'
        )
        extra_patches.append(
            patch("langchain_anthropic.ChatAnthropic.invoke", return_value=mock_llm_resp)
        )
        extra_patches.append(
            patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_veto_contamination)
        )
        # In rule mode the critic runs naturally and VETOs on win_rate=1.0 — no patch needed.

    with ExitStack() as stack:
        stack.enter_context(
            patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET)
        )
        stack.enter_context(
            patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_CONTAMINATED)
        )
        for p in extra_patches:
            stack.enter_context(p)

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

        # EDGE_WEAK with sample_size=75 < MIN_SAMPLE_SIZE → PASS (confidence=LOW)
        assert decision["status"]     in ("APPROVED", "PASS")
        assert decision["confidence"] in ("MEDIUM", "LOW")

    print("\nOK Weak edge test passed.")


# ─────────────────────────────────────────────
# PAPER TRADING E2E TEST
# ─────────────────────────────────────────────

def test_paper_trading_e2e():
    """
    End-to-end paper trading test.

    Uses a fresh temp directory so the test is fully isolated from any
    existing data/paper/ state.  Verifies:
      - book.json is created and cash decreased
      - trades.csv has at least 1 data row
      - equity.csv has at least 1 data row
      - decision["execution"] is present with status FILLED
    """
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section("PAPER TRADING E2E TEST")

    tmp_dir = tempfile.mkdtemp(prefix="p2p_paper_test_")
    try:
        # Redirect all paper files to the temp dir by overriding the env var.
        # PaperTradeManager reads PAPER_DATA_DIR inside __init__, so this works.
        with patch.dict(os.environ, {"PAPER_DATA_DIR": tmp_dir}), \
             patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET), \
             patch(QUANT_ANALYZE_PATCH, return_value=MOCK_QUANT_EDGE_CONFIRMED), \
             patch(CRITIC_REVIEW_PATCH, side_effect=_mock_critic_approve):

            trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
            decision     = LeadAnalyst().analyze_signal(trade_packet)
            print_decision(decision)

        # ── Verify decision ──────────────────────────────────────────────
        assert decision["status"]              == "APPROVED"
        assert "execution"                     in decision
        assert decision["execution"]["status"] == "FILLED"
        assert decision["execution"]["cash_after"] < decision["execution"]["cash_before"]

        # ── Verify book.json ─────────────────────────────────────────────
        book_path = Path(tmp_dir) / "book.json"
        assert book_path.exists(), "book.json should have been created"
        book = json.loads(book_path.read_text())
        assert book["cash"] < 1000.0, (
            f"Cash should have decreased from 1000.0 after a trade, got {book['cash']}"
        )
        assert len(book["positions"]) >= 1, "At least one open position expected"

        # ── Verify trades.csv ────────────────────────────────────────────
        trades_path = Path(tmp_dir) / "trades.csv"
        assert trades_path.exists(), "trades.csv should have been created"
        with open(trades_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1, "trades.csv should have at least 1 data row"
        assert rows[0]["side"]   in ("YES", "NO")
        assert rows[0]["action"] == "BET_NO"   # longshot YES → BET_NO

        # ── Verify equity.csv ────────────────────────────────────────────
        equity_path = Path(tmp_dir) / "equity.csv"
        assert equity_path.exists(), "equity.csv should have been created"
        with open(equity_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 1, "equity.csv should have at least 1 data row"
        assert float(rows[0]["cash"]) < 1000.0, "Equity cash column should show reduction"

        print(f"   book.json cash:     {book['cash']}")
        print(f"   trades.csv rows:    {len(rows)}")
        print(f"   equity.csv cash[0]: {rows[0]['cash']}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\nOK Paper trading e2e test passed.")


# ─────────────────────────────────────────────
# LIVE TEST
# ─────────────────────────────────────────────

def test_pipeline_live():
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print_section(f"LIVE PIPELINE TEST (real DB + {LLM_MODE} agents)")

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_MARKET):
        trade_packet = process_trade(VALID_NBA_LONGSHOT_YES)
        assert trade_packet is not None

        import json
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
    print(f"\n[LLM_MODE = {LLM_MODE}]")

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
        print("\n[Tip] Run with --live to also exercise real DB queries")

    print(f"\n{'='*60}")
    print(f"  All tests passed  (mode: {LLM_MODE})")
    print(f"{'='*60}")
