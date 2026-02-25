"""
src/pipeline/router.py

Classifies incoming Kalshi WebSocket trade messages by market type and
dispatches each to the appropriate handler.

Ticker prefix → market type:
  KXNBAGAME    → GAME_WINNER   longshot bias filter + full Quant/Orchestrator/Critic pipeline
  KXNBAWINS    → TOTALS        placeholder — no strategy implemented yet
  KXNBASGPROP  → PLAYER_PROP   placeholder — no strategy implemented yet
  anything else → NON_NBA / UNKNOWN (silently dropped)

Usage:
    from src.pipeline import router
    market_type, trade_packet = router.route(trade_data)
"""

from src.pipeline import bouncer


# ── Classification ────────────────────────────────────────────────────────────

def classify_market(ticker: str) -> str:
    """
    Return the market type for a given Kalshi ticker.

    Returns one of:
      "GAME_WINNER"  — KXNBAGAME-* moneyline / outright winner market
      "TOTALS"       — KXNBAWINS-* season win-total market
      "PLAYER_PROP"  — KXNBASGPROP-* player stat proposition
      "UNKNOWN"      — NBA ticker that matches none of the above prefixes
      "NON_NBA"      — non-NBA market (no "NBA" in ticker)
    """
    if ticker.startswith("KXNBAGAME"):
        return "GAME_WINNER"
    if ticker.startswith("KXNBAWINS"):
        return "TOTALS"
    if ticker.startswith("KXNBASGPROP"):
        return "PLAYER_PROP"
    if "NBA" in ticker.upper():
        return "UNKNOWN"
    return "NON_NBA"


# ── Dispatcher ────────────────────────────────────────────────────────────────

def route(trade_data: dict) -> tuple[str, dict | None]:
    """
    Classify the incoming trade and dispatch to the correct handler.

    Returns
    -------
    (market_type, trade_packet)
      market_type  : one of the strings from classify_market()
      trade_packet : a dict if the handler produced an actionable packet,
                     None if the signal was filtered out or is a placeholder.

    The caller (websocket_client) should act only on GAME_WINNER + non-None
    trade_packet pairs.  All other cases are either silently dropped (NON_NBA /
    UNKNOWN) or handled internally by the placeholder functions below.
    """
    ticker    = trade_data.get("market_ticker") or trade_data.get("ticker", "")
    yes_price = trade_data.get("yes_price")

    market_type = classify_market(ticker)

    if market_type == "GAME_WINNER":
        # Longshot bias filter + REST enrichment — full existing pipeline.
        # Returns None if the price is in the middle (no edge signal).
        packet = bouncer.process_trade(trade_data)
        return "GAME_WINNER", packet

    if market_type == "TOTALS":
        return "TOTALS", _handle_totals(ticker, yes_price)

    if market_type == "PLAYER_PROP":
        return "PLAYER_PROP", _handle_props(ticker, yes_price)

    # NON_NBA / UNKNOWN — silently drop
    return market_type, None


# ── Placeholder handlers ───────────────────────────────────────────────────────
# Each function logs the incoming signal and returns None.
# Replace the function body when a real strategy is implemented —
# the caller (websocket_client) will automatically start processing the
# returned trade_packet through the agent pipeline.

def _handle_totals(ticker: str, yes_price) -> None:
    """
    Placeholder for the Season Totals strategy.

    Season win-total markets (KXNBAWINS) are efficiently priced and driven by
    research, so the simple longshot bias rule does not apply.  A mean-reversion
    or line-movement strategy would be more appropriate here.

    TODO: implement a Totals strategy and return a trade_packet dict.
    """
    price_str = f"{yes_price}c" if yes_price is not None else "??c"
    print(f"◻ TOTALS     | {ticker:<44} | {price_str:<5}  — strategy pending")
    return None


def _handle_props(ticker: str, yes_price) -> None:
    """
    Placeholder for the Player Props strategy.

    Player prop markets (KXNBASGPROP) have a different bias profile from game
    winners.  The longshot bias erodes across seasons and varies by stat type
    (PTS30/AST10 show stronger bias than REB or lower thresholds).  A separate
    calibrated strategy is required.

    TODO: implement a Player Props strategy and return a trade_packet dict.
    """
    price_str = f"{yes_price}c" if yes_price is not None else "??c"
    print(f"◻ PLAYER_PROP| {ticker:<44} | {price_str:<5}  — strategy pending")
    return None
