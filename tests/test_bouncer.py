import pytest
from unittest.mock import patch
from src.pipeline import bouncer

# ── Mock Trades ──────────────────────────────────────────────────────────────

# ✅ Should PASS — NBA, YES is longshot (≤20¢) → action: BET_NO
mock_nba_longshot_yes = {
    "market_ticker": "KXNBAGAME-26FEB19ORLSAC-ORL",
    "yes_price": 15,
}

# ✅ Should PASS — NBA, NO is longshot (≥80¢) → action: BET_YES
mock_nba_longshot_no = {
    "market_ticker": "KXNBAGAME-26FEB19ORLSAC-SAC",
    "yes_price": 85,
}

# ❌ Should FAIL — NBA but price is in the middle (no longshot)
mock_nba_no_edge = {
    "market_ticker": "KXNBAGAME-26FEB19ORLSAC-MID",
    "yes_price": 50,
}

# ❌ Should FAIL — Not NBA
mock_non_nba = {
    "market_ticker": "NFL-2024-MAHOMES-3TD",
    "yes_price": 15,
}

# ❌ Should FAIL — Missing fields
mock_empty = {}

# ── Mock Market Details (avoids real REST calls during tests) ─────────────────
mock_market_response = {
    "title": "Will ORL win?",
    "subtitle": "Orlando Magic vs Sacramento Kings",
    "market_type": "binary",
}

# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("src.pipeline.bouncer.get_market_details", return_value=mock_market_response)
def test_nba_longshot_yes_side(mock_rest):
    """YES is the longshot (15¢) — should return BET_NO packet."""
    result = bouncer.process_trade(mock_nba_longshot_yes)
    assert result is not None, "Should have detected longshot on YES side."
    assert result["action"] == "BET_NO"
    assert result["market_price"] == 15
    assert result["category"] == "NBA"
    assert result["market_title"] == "Will ORL win?"
    print(f"✅ YES longshot: {result}")

@patch("src.pipeline.bouncer.get_market_details", return_value=mock_market_response)
def test_nba_longshot_no_side(mock_rest):
    """NO is the longshot (85¢) — should return BET_YES packet."""
    result = bouncer.process_trade(mock_nba_longshot_no)
    assert result is not None, "Should have detected longshot on NO side."
    assert result["action"] == "BET_YES"
    assert result["market_price"] == 85
    print(f"✅ NO longshot: {result}")

@patch("src.pipeline.bouncer.get_market_details", return_value=mock_market_response)
def test_nba_middle_price_rejected(mock_rest):
    """50¢ NBA trade — no longshot, should be filtered out."""
    result = bouncer.process_trade(mock_nba_no_edge)
    assert result is None, "Middle price should be rejected."
    print("✅ Middle price correctly rejected.")

@patch("src.pipeline.bouncer.get_market_details", return_value=mock_market_response)
def test_non_nba_rejected(mock_rest):
    """NFL ticker — should be silently rejected before REST call is even made."""
    result = bouncer.process_trade(mock_non_nba)
    assert result is None, "Non-NBA ticker should be rejected."
    mock_rest.assert_not_called()  # REST should never fire for non-NBA
    print("✅ Non-NBA correctly rejected without REST call.")

@patch("src.pipeline.bouncer.get_market_details", return_value=mock_market_response)
def test_empty_trade_rejected(mock_rest):
    """Empty payload — should return None without crashing."""
    result = bouncer.process_trade(mock_empty)
    assert result is None, "Empty trade should return None."
    print("✅ Empty trade correctly rejected.")

@patch("src.pipeline.bouncer.get_market_details", return_value=None)
def test_rest_failure_handled(mock_rest):
    """REST API returns None — should still return a packet with Unknown fields."""
    result = bouncer.process_trade(mock_nba_longshot_yes)
    assert result is not None, "Should still return packet even if REST fails."
    assert result["market_title"] == "Unknown"
    print(f"✅ REST failure handled gracefully: {result}")

if __name__ == "__main__":
    test_nba_longshot_yes_side()
    test_nba_longshot_no_side()
    test_nba_middle_price_rejected()
    test_non_nba_rejected()
    test_empty_trade_rejected()
    test_rest_failure_handled()