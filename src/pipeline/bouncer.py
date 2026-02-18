import duckdb
from pathlib import Path
from src.tools.kalshi_rest import get_market_details

con = duckdb.connect(database=':memory:')

def get_historical_win_rate(price, category_pattern='NBA%'):
    base_path = Path("data/kalshi")
    markets_path = base_path / "markets" / "*.parquet"
    trades_path = base_path / "trades" / "*.parquet"

    try:
        if not list(base_path.glob("markets/*.parquet")):
            return None

        query = f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM read_parquet('{markets_path}')
            WHERE status = 'finalized' 
              AND ticker LIKE '{category_pattern}'
        )
        SELECT avg(CASE WHEN t.taker_side = m.result THEN 1 ELSE 0 END) as win_rate
        FROM read_parquet('{trades_path}') t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        WHERE t.yes_price = {price}
        """
        result = con.execute(query).fetchone()
        return result[0] if result else None
    except Exception as e:
        return None

def process_trade(trade_data):
    ticker    = trade_data.get('market_ticker') or trade_data.get('ticker')
    yes_price = trade_data.get('yes_price')     or trade_data.get('price')

    if not ticker or yes_price is None:
        return None

    # --- FILTER 0: NBA only ---
    if "NBA" not in ticker.upper():
        return None

    # --- FILTER 1: Longshot Bias (both sides) ---
    is_longshot_yes = yes_price <= 20   # YES is the underdog → bet NO
    is_longshot_no  = yes_price >= 80   # NO is the underdog  → bet YES

    if not (is_longshot_yes or is_longshot_no):
        price_bar = "█" * (yes_price // 10) + "░" * (10 - yes_price // 10)
        print(f"\n❌ SKIP | {ticker:<40} | {yes_price:>3}¢ [{price_bar}] (no longshot detected)")
        return None

    # These are now guaranteed to be defined since we passed the filter above
    action = "BET_NO" if is_longshot_yes else "BET_YES"
    reason = (
        f"Longshot Bias — fading overpriced YES underdog at {yes_price}¢"
        if is_longshot_yes else
        f"Longshot Bias — fading overpriced NO underdog at {yes_price}¢"
    )

    # --- ENRICH: Fetch market details from REST API ---
    market = get_market_details(ticker)
    market_title   = market.get("title", "Unknown")       if market else "Unknown"
    market_type    = market.get("market_type", "Unknown") if market else "Unknown"
    rules_primary  = market.get("rules_primary", "")      if market else ""

    return {
        "ticker":         ticker,
        "market_price":   yes_price,
        "category":       "NBA",
        "action":         action,
        "reason":         reason,
        "market_title":   market_title,
        "market_type":    market_type,
        "rules_primary":  rules_primary,
    }