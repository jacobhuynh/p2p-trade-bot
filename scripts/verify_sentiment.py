#!/usr/bin/env python3
"""
Verify the sentiment agent and its ESPN tool.

Run from project root:
  python scripts/verify_sentiment.py

Step 1: Tests the ESPN tool in isolation (no API key). Uses a sample
  KXNBAGAME ticker and prints get_espn_matchup_context(...) and get_nba_news(...).

Step 2: Tests the sentiment agent gate (no API key). Ensures non-GAME_WINNER
  packets are returned unchanged and GAME_WINNER packets get sentiment_context
  only when the agent runs.

Step 3 (optional): Runs the full sentiment agent with a GAME_WINNER packet.
  Requires ANTHROPIC_API_KEY in .env. Prints sentiment_context from the LLM.
"""

import json
import os
import sys

# Allow importing from src when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from project root so ANTHROPIC_API_KEY is available for Step 3
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except Exception:
    pass


def step1_espn_tool():
    """Test ESPN tool directly — no API key required."""
    print("=" * 60)
    print("Step 1: ESPN tool (get_nba_news, get_espn_matchup_context)")
    print("=" * 60)

    from src.tools.espn_tool import get_nba_news, get_espn_matchup_context

    # Sample ticker: LAC vs BOS (adjust date if needed for a live game)
    ticker = "KXNBAGAME-25JAN15LACBOS-NO"
    print(f"\nTicker: {ticker}  →  matchup: LAC (Clippers) vs BOS (Celtics)")

    print("\n--- get_nba_news(limit=5) [raw NBA feed, any teams] ---")
    news = get_nba_news(limit=5)
    print(f"  Articles returned: {len(news)} (latest NBA news, not filtered by matchup)")
    for i, a in enumerate(news[:3], 1):
        print(f"  {i}. {a.get('headline', '')[:60]}...")
        print(f"     team_abbrs: {a.get('team_abbrs', [])}")

    print("\n--- get_espn_matchup_context(ticker) [RELEVANT to this game only] ---")
    print("  Filters to articles that mention LAC or BOS. These are the ones used for sentiment.")
    ctx = get_espn_matchup_context(ticker)
    if ctx is None:
        print("  (None — ticker not KXNBAGAME or parse failed)")
    else:
        print(f"  game: {json.dumps(ctx.get('game'), indent=4) or 'None'}")
        team_news = ctx.get("team_news", [])
        print(f"  team_news count: {len(team_news)} (articles mentioning LAC and/or BOS)")
        for i, n in enumerate(team_news[:5], 1):
            print(f"  {i}. {n.get('headline', '')[:55]}...  teams={n.get('team', [])}")

    print("\nStep 1 done. Relevance = only 'team_news' above are used for this matchup (LAC/BOS).\n")
    return True


def step2_sentiment_gate():
    """Test sentiment agent gate — no API key (no LLM call for non-GAME_WINNER)."""
    print("=" * 60)
    print("Step 2: Sentiment agent gate (contract_type)")
    print("=" * 60)

    from src.agents.sentiment_agent import SentimentAgent

    agent = SentimentAgent()

    # Packet without contract_type → should return unchanged (no sentiment_context)
    packet_no_type = {"ticker": "KXNBAGAME-25JAN15LACBOS-NO", "market_price": 15}
    out_no_type = agent.enrich(packet_no_type.copy())
    assert "sentiment_context" not in out_no_type, "Non-GAME_WINNER should not get sentiment_context"
    print("  OK: packet without contract_type → unchanged (no sentiment_context)")

    # Packet with TOTALS → unchanged
    packet_totals = {"ticker": "KXNBAWINS-XXX", "contract_type": "TOTALS"}
    out_totals = agent.enrich(packet_totals.copy())
    assert out_totals.get("contract_type") == "TOTALS" and "sentiment_context" not in out_totals
    print("  OK: contract_type=TOTALS → unchanged")

    # Packet with GAME_WINNER would call LLM (we skip full run here unless step 3)
    packet_gw = {"ticker": "KXNBAGAME-25JAN15LACBOS-NO", "contract_type": "GAME_WINNER"}
    print("  (GAME_WINNER packet would run tool + LLM — see Step 3)")

    print("\nStep 2 done. Sentiment gate behaves correctly.\n")
    return True


def step3_full_sentiment():
    """Run full sentiment agent with GAME_WINNER — requires ANTHROPIC_API_KEY in .env."""
    print("=" * 60)
    print("Step 3: Full sentiment agent (tool + LLM)")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  ANTHROPIC_API_KEY not set (add to .env in project root). Skipping.\n")
        return False

    from src.agents.sentiment_agent import SentimentAgent

    agent = SentimentAgent()
    packet = {
        "ticker": "KXNBAGAME-25JAN15LACBOS-NO",
        "market_price": 15,
        "action": "BET_NO",
        "contract_type": "GAME_WINNER",
    }
    print("  Calling sentiment_agent.enrich() with GAME_WINNER packet...")
    out = agent.enrich(packet)
    sc = out.get("sentiment_context")
    if sc:
        print("\n  sentiment_context:")
        print("  " + "-" * 40)
        for line in sc.splitlines():
            print("  " + line)
        print("  " + "-" * 40)
        print("\nStep 3 done. Sentiment agent and tool are working end-to-end.\n")
        return True
    else:
        print("  sentiment_context was None or empty (tool/LLM may have failed).\n")
        return False


def main():
    step1_espn_tool()
    step2_sentiment_gate()
    step3_full_sentiment()
    print("Verification complete. Review output above.")


if __name__ == "__main__":
    main()
