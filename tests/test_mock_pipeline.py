"""
tests/test_mock_pipeline.py

Mock end-to-end pipeline demo for both GAME_WINNER and PLAYER_PROP trades.

All external calls (Kalshi REST, nba_api, ESPN, Claude LLM) are mocked with
realistic data so you can see exactly what the full pipeline output looks like
without needing any API keys or live data.

Run:
    python tests/test_mock_pipeline.py               # all scenarios, full printout
    pytest tests/test_mock_pipeline.py -v -s         # same via pytest
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA — GAME WINNER
# ─────────────────────────────────────────────────────────────────────────────

MOCK_GW_MARKET = {
    "title":         "Brooklyn Nets at Cleveland Cavaliers Winner?",
    "market_type":   "binary",
    "rules_primary": "Resolves Yes if Brooklyn Nets win.",
    "open_interest": 2400,
    "volume_24h":    8100,
}

MOCK_GW_TRADE = {
    "market_ticker": "KXNBAGAME-25APR21BKNCLE-BKN",
    "yes_price": 14,
}

MOCK_GW_QUANT_APPROVED = {
    "historical_edge":          0.07,
    "actual_win_rate":          0.92,
    "implied_prob":             0.85,
    "no_win_rate":              0.90,
    "yes_no_asymmetry":         0.10,
    "sample_size":              418,
    "data_quality":             "SUFFICIENT",
    "verdict":                  "EDGE_CONFIRMED",
    "calibration_gap":          0.07,
    "summary":                  (
        "Strong longshot bias at 14¢: historical NO win rate of 92% vs. implied 85%, "
        "a 7pp calibration gap across 418 samples. "
        "Cleveland is listed as STATUS_SCHEDULED with no injury concerns surfaced by ESPN; "
        "both teams are fresh off a two-day rest. "
        "Sample size of 418 is solid, though the edge has narrowed slightly in recent seasons — "
        "worth monitoring for regime change."
    ),
    "game_context": {
        "home_abbr": "CLE", "away_abbr": "BKN",
        "status": "STATUS_SCHEDULED",
        "home_score": 0, "away_score": 0, "winner_abbr": None,
    },
    "team_stats": {
        "home": {"abbr": "CLE", "last10": "7-3", "home_record": "4-1", "away_record": "3-2"},
        "away": {"abbr": "BKN", "last10": "2-8", "home_record": "1-4", "away_record": "1-4"},
    },
    "home_key_players": [
        {"name": "Donovan Mitchell", "avg_pts": 26.4, "avg_reb": 4.9, "avg_ast": 5.8, "last5_pts": [29, 24, 31, 22, 28]},
        {"name": "Darius Garland",   "avg_pts": 19.1, "avg_reb": 2.8, "avg_ast": 6.4, "last5_pts": [21, 17, 24, 18, 15]},
        {"name": "Evan Mobley",      "avg_pts": 16.2, "avg_reb": 9.1, "avg_ast": 2.8, "last5_pts": [14, 18, 12, 20, 17]},
    ],
    "away_key_players": [
        {"name": "Cam Thomas",    "avg_pts": 21.3, "avg_reb": 3.1, "avg_ast": 3.2, "last5_pts": [19, 23, 17, 25, 22]},
        {"name": "Nic Claxton",   "avg_pts": 12.1, "avg_reb": 8.4, "avg_ast": 2.1, "last5_pts": [11, 14, 9,  13, 12]},
        {"name": "Dennis Schroder","avg_pts": 11.8, "avg_reb": 2.4, "avg_ast": 5.1, "last5_pts": [10, 8,  14, 12, 11]},
    ],
    "orderbook_depth_at_price": 87,
    "price_bucket_edge":  {"actual_win_rate": 0.92, "edge": 0.07, "sample_size": 418},
    "longshot_bias":      {"no_win_rate": 0.90, "avg_price": 12.4, "sample_size": 2103},
    "taker_win_rate":     {"win_rate": 0.91, "sample_size": 412},
    "inverse_bucket":     {"actual_win_rate": 0.08, "edge": -0.07, "sample_size": 389},
}

MOCK_GW_QUANT_VETOED = {
    **MOCK_GW_QUANT_APPROVED,
    "actual_win_rate":  1.0,
    "no_win_rate":      1.0,
    "calibration_gap":  0.15,
    "verdict":          "EDGE_CONFIRMED",
    "summary": (
        "Calibration gap of 15pp looks extremely strong, but the actual_win_rate of 1.0 across 418 samples "
        "is a red flag — no real market produces a perfect win rate at scale. "
        "ESPN shows no injury news for either team, and the game is scheduled normally. "
        "The suspiciously perfect win rate strongly suggests data contamination or a market type mismatch "
        "in the historical parquet files."
    ),
    "price_bucket_edge": {"actual_win_rate": 1.0, "edge": 0.15, "sample_size": 418},
}

MOCK_GW_CRITIC_APPROVE = {
    "decision":       "APPROVE",
    "veto_reason":    None,
    "concerns":       ["Sample size of 418 is adequate but edge has narrowed in recent seasons"],
    "risk_score":     3,
    "summary":        "Solid longshot bias play — 7pp calibration gap, clean data, sufficient liquidity.",
    "sentiment_note": "ESPN shows game on schedule with no injury flags; sentiment aligns with the edge.",
    "status":         "APPROVED",
}

MOCK_GW_CRITIC_VETO = {
    "decision":       "VETO",
    "veto_reason":    "actual_win_rate of 1.0 across 418 samples is statistically impossible — indicates data contamination, not genuine edge.",
    "concerns":       ["Win rate of 1.0 across hundreds of samples is a red flag", "Calibration gap of 15pp is implausibly large"],
    "risk_score":     9,
    "summary":        "Data contamination detected — perfect win rate is a hard veto.",
    "sentiment_note": "Sentiment was normal (no injuries, game scheduled), but data quality overrides everything here.",
    "status":         "VETOED",
}

MOCK_SENTIMENT_GW = (
    "Cleveland Cavaliers are 7-3 in their last 10 and host Brooklyn tonight. "
    "No injury concerns for Donovan Mitchell or Darius Garland per ESPN. "
    "Brooklyn listed Cam Thomas as questionable (ankle) but he's expected to play. "
    "Cleveland has won 4 straight home games against bottom-5 Eastern Conference opponents."
)

# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA — PLAYER PROP
# ─────────────────────────────────────────────────────────────────────────────

MOCK_PROP_MARKET = {
    "title":         "Will LeBron James score 25+ points?",
    "market_type":   "binary",
    "rules_primary": "Resolves Yes if LeBron James scores 25 or more points in the game.",
    "open_interest": 1800,
    "volume_24h":    5200,
}

MOCK_PROP_TRADE = {
    "market_ticker": "KXNBASGPROP-25APR21LEBRON-PTS25",
    "yes_price": 18,
}

MOCK_PROP_AGENT_APPROVED = {
    "player_name":    "LeBron James",
    "prop_type":      "PTS",
    "prop_threshold": 25.0,
    "action":         "BET_NO",
    "price":          18,
    "side":           "no",
    "ticker":         "KXNBASGPROP-25APR21LEBRON-PTS25",
    "recent_avg":     29.4,
    "hit_rate":       0.80,
    "variance":       4.2,
    "edge":           4.4,
    "n_games_sampled": 10,
    "verdict":        "EDGE_CONFIRMED",
    "data_quality":   "SUFFICIENT",
    "confidence":     "HIGH",
    "kelly_fraction": 0.04,
    "summary": (
        "LeBron has exceeded 25 points in 8 of his last 10 games with a 29.4 average — "
        "the prop line at 18¢ (implied 18% YES probability) is badly mispriced against this trend. "
        "He's averaged 31.2 points vs. Boston over 5 recent matchups, suggesting elevated "
        "motivation in this rivalry game. "
        "Variance of 4.2 is moderate; the main risk is load management given back-to-back "
        "scheduling, though no rest designation has been announced."
    ),
    "matchup_context": {
        "avg_pts_vs_opp": 31.2, "avg_reb_vs_opp": 8.4,
        "avg_ast_vs_opp": 7.1, "n_games": 5,
    },
    "usage_rate": {"usage_rate": 0.312, "ts_pct": 0.601, "pace": 100.4},
    "quant_summary": {
        "calibration_gap": 4.4, "actual_win_rate": 0.80,
        "sample_size": 10, "verdict": "EDGE_CONFIRMED",
    },
}

MOCK_PROP_AGENT_VETOED = {
    **MOCK_PROP_AGENT_APPROVED,
    "hit_rate":       0.95,
    "verdict":        "EDGE_CONFIRMED",
    "summary": (
        "LeBron has exceeded 25 points in 9.5 of 10 recent games — suspiciously high hit rate "
        "that warrants data quality review. "
        "Matchup vs. Boston is favorable historically at 31.2 ppg average. "
        "The 95% hit rate is implausibly consistent and may reflect a cherry-picked sample or "
        "a period when the prop line was unusually low."
    ),
    "quant_summary": {
        "calibration_gap": 4.4, "actual_win_rate": 0.95,
        "sample_size": 10, "verdict": "EDGE_CONFIRMED",
        "hit_rate": 0.95, "variance": 1.8,
        "recent_avg": 29.4, "prop_threshold": 25.0,
        "n_games_sampled": 10, "matchup_context": None,
    },
}

MOCK_PROP_AGENT_PASS = {
    **MOCK_PROP_AGENT_APPROVED,
    "recent_avg":     22.1,
    "hit_rate":       0.40,
    "edge":           -2.9,
    "verdict":        "NO_EDGE",
    "confidence":     "LOW",
    "kelly_fraction": 0.01,
    "summary": (
        "LeBron has exceeded 25 points in only 4 of his last 10 games with a 22.1 average — "
        "the prop line at 18¢ actually appears fairly priced given this recent slump. "
        "He's averaging 24.0 vs. Boston over recent matchups, just under the threshold. "
        "No edge detected; the market appears to have already priced in his recent form."
    ),
    "quant_summary": {
        "calibration_gap": -2.9, "actual_win_rate": 0.40,
        "sample_size": 10, "verdict": "NO_EDGE",
    },
}

MOCK_PROP_CRITIC_APPROVE = {
    "decision":       "APPROVE",
    "veto_reason":    None,
    "concerns":       ["Only 10 games sampled — larger sample would increase confidence"],
    "risk_score":     4,
    "summary":        "Strong hit rate on a well-established trend; player news confirms availability.",
    "sentiment_note": "Injury news confirms LeBron is active tonight; no usage restriction flagged.",
    "status":         "APPROVED",
}

MOCK_PROP_CRITIC_VETO = {
    "decision":       "VETO",
    "veto_reason":    "hit_rate of 0.95 across 10 games is implausibly consistent for a stat-based prop — likely overfitted on a short sample or cherry-picked window.",
    "concerns":       ["95% hit rate is a data quality red flag", "n_games_sampled = 10 is borderline for EDGE_CONFIRMED"],
    "risk_score":     8,
    "summary":        "Suspiciously high hit rate on a small sample — vetoing on data quality grounds.",
    "sentiment_note": "Player availability is not in question, but statistical reliability is.",
    "status":         "VETOED",
}

MOCK_SENTIMENT_PROP = (
    "LeBron James is listed as active and expected to play full minutes tonight per ESPN. "
    "No injury or load management designation as of this afternoon. "
    "Anthony Davis is also active, meaning LeBron's usage rate should remain consistent "
    "at his season average of 31.2% without defensive scheming adjustments."
)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS  (mirrors websocket_client.py output format)
# ─────────────────────────────────────────────────────────────────────────────

def _print_gw_decision(decision, trade_packet):
    status = decision.get("status")
    quant  = decision.get("quant_summary", {})
    critic = decision.get("critic", {})
    ts     = datetime.now().strftime("%H:%M:%S")
    emoji  = {"APPROVED": "✅", "VETOED": "🚫", "PASS": "⏭️"}.get(status, "❓")

    print(f"\n{'='*60}")
    print(f"🚨 GAME WINNER | {ts}  {emoji} {status}")
    print(f"{'='*60}")
    print(f"📌 Ticker:      {trade_packet.get('ticker') or trade_packet.get('market_ticker')}")
    print(f"📋 Title:       {trade_packet.get('market_title', 'N/A')}")
    print(f"💰 Price:       {decision.get('price')}¢")
    print(f"🎯 Action:      {decision.get('action')}")
    print(f"{'─'*60}")
    sc = decision.get("sentiment_context")
    print(f"📰 Sentiment (live ESPN context)")
    if sc:
        for line in sc.strip().splitlines():
            print(f"   {line}")
    else:
        print(f"   (no data)")
    print(f"{'─'*60}")
    print(f"📊 Quant")
    print(f"   Gap:           {quant.get('calibration_gap')}")
    print(f"   Win Rate:      {quant.get('actual_win_rate')}")
    print(f"   Implied Prob:  {quant.get('implied_prob')}")
    print(f"   Sample:        {quant.get('sample_size')}  [{quant.get('data_quality')}]")
    print(f"   Verdict:       {quant.get('verdict')}")
    print(f"   NO Win Rate:   {quant.get('no_win_rate')}  (longshot bias)")
    print(f"   Asymmetry:     {quant.get('yes_no_asymmetry')}")
    pb  = quant.get("price_bucket_edge") or {}
    lb  = quant.get("longshot_bias") or {}
    tw  = quant.get("taker_win_rate") or {}
    inv = quant.get("inverse_bucket") or {}
    print(f"   Price Bucket:  win={pb.get('actual_win_rate')}  edge={pb.get('edge')}  n={pb.get('sample_size')}")
    print(f"   Taker WR:      win={tw.get('win_rate')}  n={tw.get('sample_size')}")
    print(f"   Longshot Bias: no_wr={lb.get('no_win_rate')}  avg_price={lb.get('avg_price')}  n={lb.get('sample_size')}")
    print(f"   Inverse Bkt:   win={inv.get('actual_win_rate')}  edge={inv.get('edge')}  n={inv.get('sample_size')}")
    gc = quant.get("game_context")
    if gc:
        score_str = f"{gc.get('away_abbr')} {gc.get('away_score')} @ {gc.get('home_abbr')} {gc.get('home_score')}"
        status_str = gc.get("status", "").replace("STATUS_", "")
        print(f"   ESPN:          {score_str}  [{status_str}]")
    ts_dict = quant.get("team_stats")
    if ts_dict:
        h = ts_dict.get("home") or {}
        a = ts_dict.get("away") or {}
        print(f"   NBA Records:   {h.get('abbr')} last10={h.get('last10')} home={h.get('home_record')} away={h.get('away_record')}")
        print(f"                  {a.get('abbr')} last10={a.get('last10')} home={a.get('home_record')} away={a.get('away_record')}")
    for label, players in [("home", quant.get("home_key_players")), ("away", quant.get("away_key_players"))]:
        if players:
            for p in players:
                l5 = ", ".join(str(x) for x in p.get("last5_pts", []))
                print(f"   {label.upper()} key:      {p['name']}  {p['avg_pts']}pts/{p['avg_reb']}reb/{p['avg_ast']}ast  last5=[{l5}]")
    print(f"   Summary:       {quant.get('summary')}")
    print(f"{'─'*60}")
    print(f"🧠 Orchestrator")
    print(f"   Confidence:  {decision.get('confidence')}")
    print(f"   Edge:        {decision.get('edge')}")
    print(f"   Kelly:       {decision.get('kelly_fraction')}")
    print(f"   Reason:      {decision.get('reason')}")
    if critic:
        print(f"{'─'*60}")
        print(f"🔍 Critic")
        print(f"   Decision:    {critic.get('decision')}")
        print(f"   Risk Score:  {critic.get('risk_score')}/10")
        if critic.get("veto_reason"):
            print(f"   Veto:        {critic.get('veto_reason')}")
        for c in (critic.get("concerns") or []):
            print(f"   ⚠️  {c}")
        print(f"   Summary:     {critic.get('summary')}")
        print(f"   Sentiment:   {critic.get('sentiment_note', '')}")
    if status == "APPROVED":
        print(f"{'─'*60}")
        print(f"💾 [MOCK] Would log as trade — run `python -m src.settle` to check P&L")
    print(f"{'='*60}")


def _print_prop_decision(decision, trade_packet):
    status = decision.get("status")
    quant  = decision.get("quant_summary", {})
    critic = decision.get("critic", {})
    ts     = datetime.now().strftime("%H:%M:%S")
    emoji  = {"APPROVED": "✅", "VETOED": "🚫", "PASS": "⏭️"}.get(status, "❓")
    player = decision.get("player_name") or trade_packet.get("player_name", "?")
    prop_t = decision.get("prop_type") or trade_packet.get("prop_type", "?")
    thresh = decision.get("prop_threshold") or trade_packet.get("prop_threshold", "?")

    print(f"\n{'='*60}")
    print(f"🏀 PLAYER PROP | {ts}  {emoji} {status}")
    print(f"{'='*60}")
    print(f"📌 Ticker:      {trade_packet.get('ticker') or trade_packet.get('market_ticker')}")
    print(f"👤 Player:      {player}  |  {prop_t} {thresh}+")
    print(f"🎯 Action:      {decision.get('action')}  @ {decision.get('price')}¢")
    print(f"{'─'*60}")
    sc = decision.get("sentiment_context")
    print(f"📰 Sentiment (player news)")
    if sc:
        for line in sc.strip().splitlines():
            print(f"   {line}")
    else:
        print(f"   (no player news found)")
    print(f"{'─'*60}")
    print(f"📊 Prop Stats")
    print(f"   Hit Rate:    {quant.get('actual_win_rate') or quant.get('hit_rate')}")
    print(f"   Avg vs Line: {quant.get('recent_avg')} vs {quant.get('prop_threshold')}")
    print(f"   Edge:        {quant.get('calibration_gap')}")
    print(f"   Variance:    {quant.get('variance')}")
    print(f"   Sample:      {quant.get('sample_size') or quant.get('n_games_sampled')} games  [{quant.get('data_quality')}]")
    print(f"   Verdict:     {quant.get('verdict')}")
    mc = quant.get("matchup_context")
    if mc:
        print(f"   vs Opp:      pts={mc.get('avg_pts_vs_opp')} reb={mc.get('avg_reb_vs_opp')} ast={mc.get('avg_ast_vs_opp')}  (n={mc.get('n_games')})")
    print(f"   Summary:     {quant.get('summary')}")
    print(f"{'─'*60}")
    print(f"🧠 Orchestrator")
    print(f"   Confidence:  {decision.get('confidence')}")
    print(f"   Edge:        {decision.get('edge')}")
    print(f"   Kelly:       {decision.get('kelly_fraction')}")
    print(f"   Reason:      {decision.get('reason')}")
    if critic:
        print(f"{'─'*60}")
        print(f"🔍 Critic")
        print(f"   Decision:    {critic.get('decision')}")
        print(f"   Risk Score:  {critic.get('risk_score')}/10")
        if critic.get("veto_reason"):
            print(f"   Veto:        {critic.get('veto_reason')}")
        for c in (critic.get("concerns") or []):
            print(f"   ⚠️  {c}")
        print(f"   Summary:     {critic.get('summary')}")
        print(f"   Sentiment:   {critic.get('sentiment_note', '')}")
    if status == "APPROVED":
        print(f"{'─'*60}")
        print(f"💾 [MOCK] Would log as trade — settlement requires prop result entry")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# GAME WINNER TESTS
# ─────────────────────────────────────────────────────────────────────────────

PROP_AGENT_PATCH  = "src.agents.prop_agent.PlayerPropAgent.analyze"
QUANT_AGENT_PATCH = "src.agents.game_quant_agent.QuantAgent.analyze"
CRITIC_PATCH      = "src.agents.critic.CriticAgent.review"
SENTIMENT_PATCH   = "src.agents.sentiment_agent.SentimentAgent.enrich"


def _sentiment_side_effect_gw(text):
    def _enrich(packet):
        packet["sentiment_context"] = text
        return packet
    return _enrich


def _sentiment_side_effect_prop(text):
    def _enrich(packet):
        packet["sentiment_context"] = text
        return packet
    return _enrich


def test_game_winner_approved():
    """BKN at 14¢ — strong longshot edge, Critic approves."""
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 1 — GAME WINNER  ✅ APPROVED")
    print(f"  BKN at 14¢ — 7pp calibration gap, sample=418, Critic approves")
    print(f"{'#'*60}")

    def _critic_approve(tp, d):
        return {**d, "status": "APPROVED", "action": d.get("action"), "critic": MOCK_GW_CRITIC_APPROVE}

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_GW_MARKET), \
         patch(QUANT_AGENT_PATCH, return_value=MOCK_GW_QUANT_APPROVED), \
         patch(SENTIMENT_PATCH, side_effect=_sentiment_side_effect_gw(MOCK_SENTIMENT_GW)), \
         patch(CRITIC_PATCH, side_effect=_critic_approve):

        trade_packet = process_trade(MOCK_GW_TRADE)
        assert trade_packet is not None, "Bouncer should pass a 14¢ longshot"
        decision = LeadAnalyst().analyze_signal(trade_packet)

    _print_gw_decision({**decision, "quant_summary": MOCK_GW_QUANT_APPROVED}, trade_packet)

    assert decision["status"]             == "APPROVED"
    assert decision["action"]             == "BET_NO"
    assert decision["critic"]["decision"] == "APPROVE"
    assert decision["critic"]["risk_score"] <= 5
    print("OK assertions passed")


def test_game_winner_vetoed():
    """BKN at 14¢ — data contamination (win_rate=1.0), Critic vetoes."""
    from src.pipeline.bouncer import process_trade
    from src.agents.orchestrator import LeadAnalyst

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 2 — GAME WINNER  🚫 VETOED")
    print(f"  BKN at 14¢ — win_rate=1.0, data contamination, Critic vetoes")
    print(f"{'#'*60}")

    def _critic_veto(tp, d):
        return {**d, "status": "VETOED", "action": "PASS", "critic": MOCK_GW_CRITIC_VETO}

    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_GW_MARKET), \
         patch(QUANT_AGENT_PATCH, return_value=MOCK_GW_QUANT_VETOED), \
         patch(SENTIMENT_PATCH, side_effect=_sentiment_side_effect_gw(MOCK_SENTIMENT_GW)), \
         patch(CRITIC_PATCH, side_effect=_critic_veto):

        trade_packet = process_trade(MOCK_GW_TRADE)
        decision = LeadAnalyst().analyze_signal(trade_packet)

    _print_gw_decision({**decision, "quant_summary": MOCK_GW_QUANT_VETOED}, trade_packet)

    assert decision["status"]             == "VETOED"
    assert decision["critic"]["decision"] == "VETO"
    assert "contamination" in decision["critic"]["veto_reason"].lower() or "1.0" in decision["critic"]["veto_reason"]
    print("OK assertions passed")


def test_game_winner_pass():
    """BKN at 55¢ — mid-price, bouncer filters before pipeline runs."""
    from src.pipeline.bouncer import process_trade
    from src.pipeline.router import route

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 3 — GAME WINNER  ⏭️  PASS (bouncer drop)")
    print(f"  BKN at 55¢ — no longshot detected, filtered before pipeline")
    print(f"{'#'*60}")

    mid_price_trade = {"market_ticker": "KXNBAGAME-25APR21BKNCLE-BKN", "yes_price": 55}
    with patch("src.pipeline.bouncer.get_market_details", return_value=MOCK_GW_MARKET):
        market_type, trade_packet = route(mid_price_trade)

    print(f"\n   market_type:  {market_type}")
    print(f"   trade_packet: {trade_packet}  (None = bouncer dropped it)")
    print(f"   ⏭️  Skipped — printing dot in the live bot")

    assert market_type   == "GAME_WINNER"
    assert trade_packet  is None
    print("OK assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER PROP TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_prop_approved():
    """LeBron 25+ pts at 18¢ — 80% hit rate, Critic approves."""
    from src.agents.orchestrator import LeadAnalyst

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 4 — PLAYER PROP  ✅ APPROVED")
    print(f"  LeBron 25+ PTS at 18¢ — hit_rate=0.80, avg=29.4, Critic approves")
    print(f"{'#'*60}")

    def _critic_approve(tp, d):
        return {**d, "status": "APPROVED", "critic": MOCK_PROP_CRITIC_APPROVE}

    trade_packet = {
        "ticker":          "KXNBASGPROP-25APR21LEBRON-PTS25",
        "market_price":    18,
        "action":          "BET_NO",
        "contract_type":   "PLAYER_PROP",
        "market_title":    "Will LeBron James score 25+ points?",
        "player_name":     "LeBron James",
        "prop_type":       "PTS",
        "prop_threshold":  25.0,
        "opponent_abbr":   "BOS",
        "live_open_interest": 1800,
    }

    with patch(PROP_AGENT_PATCH, return_value=MOCK_PROP_AGENT_APPROVED), \
         patch(SENTIMENT_PATCH, side_effect=_sentiment_side_effect_prop(MOCK_SENTIMENT_PROP)), \
         patch(CRITIC_PATCH, side_effect=_critic_approve):

        decision = LeadAnalyst().analyze_prop_signal(trade_packet)

    full_quant = {
        **decision.get("quant_summary", {}),
        "summary":         MOCK_PROP_AGENT_APPROVED["summary"],
        "matchup_context": MOCK_PROP_AGENT_APPROVED["matchup_context"],
        "data_quality":    "SUFFICIENT",
    }
    _print_prop_decision({**decision, "quant_summary": full_quant}, trade_packet)

    assert decision["status"]             == "APPROVED"
    assert decision["critic"]["decision"] == "APPROVE"
    assert decision["critic"]["risk_score"] <= 5
    print("OK assertions passed")


def test_prop_vetoed():
    """LeBron 25+ pts — hit_rate=0.95, suspiciously high, Critic vetoes."""
    from src.agents.orchestrator import LeadAnalyst

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 5 — PLAYER PROP  🚫 VETOED")
    print(f"  LeBron 25+ PTS — hit_rate=0.95 (implausibly consistent), Critic vetoes")
    print(f"{'#'*60}")

    def _critic_veto(tp, d):
        return {**d, "status": "VETOED", "action": "PASS", "critic": MOCK_PROP_CRITIC_VETO}

    trade_packet = {
        "ticker":          "KXNBASGPROP-25APR21LEBRON-PTS25",
        "market_price":    18,
        "action":          "BET_NO",
        "contract_type":   "PLAYER_PROP",
        "market_title":    "Will LeBron James score 25+ points?",
        "player_name":     "LeBron James",
        "prop_type":       "PTS",
        "prop_threshold":  25.0,
        "opponent_abbr":   "BOS",
    }

    with patch(PROP_AGENT_PATCH, return_value=MOCK_PROP_AGENT_VETOED), \
         patch(SENTIMENT_PATCH, side_effect=_sentiment_side_effect_prop(MOCK_SENTIMENT_PROP)), \
         patch(CRITIC_PATCH, side_effect=_critic_veto):

        decision = LeadAnalyst().analyze_prop_signal(trade_packet)

    full_quant = {
        **decision.get("quant_summary", {}),
        "summary":      MOCK_PROP_AGENT_VETOED["summary"],
        "data_quality": "SUFFICIENT",
    }
    _print_prop_decision({**decision, "quant_summary": full_quant}, trade_packet)

    assert decision["status"]             == "VETOED"
    assert decision["critic"]["decision"] == "VETO"
    assert decision["critic"]["risk_score"] >= 7
    print("OK assertions passed")


def test_prop_pass_no_edge():
    """LeBron 25+ pts — hit_rate=0.40, Python gate passes before Critic."""
    from src.agents.orchestrator import LeadAnalyst

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 6 — PLAYER PROP  ⏭️  PASS (no edge)")
    print(f"  LeBron 25+ PTS — hit_rate=0.40, avg=22.1, gate passes before Critic")
    print(f"{'#'*60}")

    trade_packet = {
        "ticker":          "KXNBASGPROP-25APR21LEBRON-PTS25",
        "market_price":    18,
        "action":          "BET_NO",
        "contract_type":   "PLAYER_PROP",
        "market_title":    "Will LeBron James score 25+ points?",
        "player_name":     "LeBron James",
        "prop_type":       "PTS",
        "prop_threshold":  25.0,
    }

    with patch(PROP_AGENT_PATCH, return_value=MOCK_PROP_AGENT_PASS), \
         patch(SENTIMENT_PATCH, side_effect=_sentiment_side_effect_prop(MOCK_SENTIMENT_PROP)) as mock_sentiment, \
         patch(CRITIC_PATCH) as mock_critic:

        decision = LeadAnalyst().analyze_prop_signal(trade_packet)

    print(f"\n   Status:     {decision.get('status')}")
    print(f"   Reason:     {decision.get('reason')}")
    print(f"   Edge:       {decision.get('edge')}")
    print(f"   Confidence: {decision.get('confidence')}")
    print(f"   ⏭️  Critic never called — gate stopped it in Python")
    print(f"   Summary:    {MOCK_PROP_AGENT_PASS['summary']}")

    assert decision["status"] == "PASS"
    assert decision.get("critic") is None
    mock_critic.assert_not_called()
    print("OK assertions passed — Critic was never called")


# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOGGER MOCK E2E
# ─────────────────────────────────────────────────────────────────────────────

def test_trade_logger_both_market_types():
    """Log one GAME_WINNER and one PLAYER_PROP trade; verify market_type column."""
    from src.execution.trade_logger import TradeLogger

    print(f"\n\n{'#'*60}")
    print(f"  SCENARIO 7 — TRADE LOGGER  both market types")
    print(f"{'#'*60}")

    tmp_db = tempfile.mktemp(prefix="mock_pipeline_test_", suffix=".db")
    try:
        logger = TradeLogger(db_path=tmp_db)

        gw_decision = {
            "action": "BET_NO", "side": "no", "price": 14,
            "kelly_fraction": 0.04, "confidence": "HIGH",
            "ticker": "KXNBAGAME-25APR21BKNCLE-BKN",
            "status": "APPROVED",
            "quant_summary": MOCK_GW_QUANT_APPROVED,
            "critic": MOCK_GW_CRITIC_APPROVE,
        }
        gw_packet = {
            "ticker": "KXNBAGAME-25APR21BKNCLE-BKN",
            "market_title": "Brooklyn Nets at Cleveland Cavaliers Winner?",
            "contract_type": "GAME_WINNER",
        }
        gw_id = logger.log_trade(gw_decision, gw_packet)

        prop_decision = {
            "action": "BET_NO", "side": "no", "price": 18,
            "kelly_fraction": 0.04, "confidence": "HIGH",
            "ticker": "KXNBASGPROP-25APR21LEBRON-PTS25",
            "status": "APPROVED",
            "quant_summary": MOCK_PROP_AGENT_APPROVED["quant_summary"],
            "critic": MOCK_PROP_CRITIC_APPROVE,
        }
        prop_packet = {
            "ticker": "KXNBASGPROP-25APR21LEBRON-PTS25",
            "market_title": "Will LeBron James score 25+ points?",
            "contract_type": "PLAYER_PROP",
            "player_name":   "LeBron James",
            "prop_threshold": 25.0,
        }
        prop_id = logger.log_trade(prop_decision, prop_packet)

        open_trades = logger.open_trades()
        assert len(open_trades) == 2

        gw_row   = next(r for r in open_trades if r["id"] == gw_id)
        prop_row = next(r for r in open_trades if r["id"] == prop_id)

        assert gw_row["market_type"]   == "GAME_WINNER"
        assert prop_row["market_type"] == "PLAYER_PROP"
        assert prop_row["player_name"] == "LeBron James"
        assert prop_row["prop_threshold"] == 25.0

        print(f"\n   Trade #{gw_id}:   {gw_row['ticker']:<44}  market_type={gw_row['market_type']}")
        print(f"   Trade #{prop_id}:   {prop_row['ticker']:<44}  market_type={prop_row['market_type']}  player={prop_row['player_name']}  line={prop_row['prop_threshold']}")
        print(f"\n   Both status=PENDING_RESOLUTION — run `python -m src.settle` to evaluate")

    finally:
        Path(tmp_db).unlink(missing_ok=True)

    print("OK assertions passed")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_game_winner_approved()
    test_game_winner_vetoed()
    test_game_winner_pass()
    test_prop_approved()
    test_prop_vetoed()
    test_prop_pass_no_edge()
    test_trade_logger_both_market_types()

    print(f"\n\n{'='*60}")
    print(f"  All 7 mock pipeline scenarios passed")
    print(f"  No API keys required — all external calls were mocked")
    print(f"{'='*60}")
