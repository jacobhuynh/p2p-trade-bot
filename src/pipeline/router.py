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
    if ticker.startswith(("KXNBAPTS", "KXNBASGPROP")):
        return "PLAYER_PROP"
    if ticker.startswith(("KXNBA2D", "KXNBA3D", "KXNBASPREAD", "KXNBATOTAL", "KXNBA1HTOTAL", "KXNBASERIES")):
        return "UNKNOWN"
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
    if (p := trade_data.get("yes_price_dollars")) is not None:
        yes_price = round(float(p) * 100)
    elif (p := trade_data.get("yes_price")) is not None:
        yes_price = int(p)
    else:
        yes_price = None

    market_type = classify_market(ticker)

    if market_type == "GAME_WINNER":
        packet = bouncer.process_trade(trade_data)
        if packet is not None:
            packet["contract_type"] = "GAME_WINNER"
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


def _handle_props(ticker: str, yes_price) -> dict | None:
    """
    Player Props strategy handler.

    Fetches market details from the Kalshi REST API to extract player name,
    prop type, and threshold from the market title/rules.  Applies a longshot
    bias filter (same ≤20 / ≥80 boundaries as game winners) before returning
    an actionable trade packet for PlayerPropAgent.
    """
    from src.tools.kalshi_rest import get_market_details

    price_str = f"{yes_price}c" if yes_price is not None else "??c"

    market = get_market_details(ticker)
    if market is None:
        print(f"◻ PLAYER_PROP| {ticker:<44} | {price_str:<5}  — market details unavailable")
        return None

    title = market.get("title", "")
    rules = market.get("rules_primary", "")

    parsed = _parse_prop_from_market(title, rules)
    if parsed is None:
        print(f"◻ PLAYER_PROP| {ticker:<44} | {price_str:<5}  — could not parse prop details")
        return None

    player_name, prop_type, prop_threshold = parsed

    if yes_price is None:
        print(f"◻ PLAYER_PROP| {ticker:<44}  — no price")
        return None

    is_longshot_yes = yes_price <= 20
    is_longshot_no  = yes_price >= 80
    if not (is_longshot_yes or is_longshot_no):
        print(f"◻ PLAYER_PROP| {ticker:<44} | {price_str:<5}  — no longshot signal")
        return None

    action = "BET_NO" if is_longshot_yes else "BET_YES"

    return {
        "ticker":             ticker,
        "market_price":       yes_price,
        "action":             action,
        "market_title":       title,
        "contract_type":      "PLAYER_PROP",
        "player_name":        player_name,
        "prop_type":          prop_type,
        "prop_threshold":     prop_threshold,
        "live_open_interest": market.get("open_interest", 0),
    }


def _parse_prop_from_market(title: str, rules: str) -> tuple | None:
    """
    Extract (player_name, prop_type, threshold) from a Kalshi market title.

    Handles common patterns:
      "Will LeBron James score 25+ points?"      → ("LeBron James", "PTS", 25.0)
      "Will Anthony Davis record 10+ rebounds?"  → ("Anthony Davis", "REB", 10.0)
      "Will Nikola Jokic record 10+ assists?"    → ("Nikola Jokic", "AST", 10.0)

    Returns None if the title does not match any known pattern.
    """
    import re as _re

    patterns = [
        # "Jaylon Tyson: 15+ points"
        (_re.compile(r"^(.+?):\s*(\d+(?:\.\d+)?)\+?\s+points?$", _re.IGNORECASE), "PTS"),
        (_re.compile(r"^(.+?):\s*(\d+(?:\.\d+)?)\+?\s+rebounds?$", _re.IGNORECASE), "REB"),
        (_re.compile(r"^(.+?):\s*(\d+(?:\.\d+)?)\+?\s+assists?$", _re.IGNORECASE), "AST"),
        # "Will X score 15+ points?"
        (_re.compile(r"Will (.+?)\s+score\s+(\d+(?:\.\d+)?)\+?\s+points?", _re.IGNORECASE), "PTS"),
        (_re.compile(r"Will (.+?)\s+record\s+(\d+(?:\.\d+)?)\+?\s+rebounds?", _re.IGNORECASE), "REB"),
        (_re.compile(r"Will (.+?)\s+record\s+(\d+(?:\.\d+)?)\+?\s+assists?", _re.IGNORECASE), "AST"),
        (_re.compile(r"Will (.+?)\s+have\s+(\d+(?:\.\d+)?)\+?\s+points?", _re.IGNORECASE), "PTS"),
        (_re.compile(r"Will (.+?)\s+have\s+(\d+(?:\.\d+)?)\+?\s+rebounds?", _re.IGNORECASE), "REB"),
        (_re.compile(r"Will (.+?)\s+have\s+(\d+(?:\.\d+)?)\+?\s+assists?", _re.IGNORECASE), "AST"),
    ]

    for text in (title, rules):
        for pattern, prop_type in patterns:
            m = pattern.search(text)
            if m:
                player_name = m.group(1).strip().rstrip("?").strip()
                threshold   = float(m.group(2))
                return player_name, prop_type, threshold

    return None
