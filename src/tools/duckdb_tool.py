"""
src/tools/duckdb_tool.py

Low-level DuckDB query functions against the parquet data store.
Called by quant.py — never called directly by agents.
"""

import duckdb
from pathlib import Path

# Shared connection — read-only against parquet files
_DB_PATH = Path("data/kalshi")
_MARKETS = str(_DB_PATH / "markets" / "*.parquet")
_TRADES  = str(_DB_PATH / "trades"  / "*.parquet")


def _con():
    return duckdb.connect(database=":memory:")


def get_historical_win_rate(price: int, category_pattern: str = "KXNBA%") -> dict:
    """
    At this exact yes_price, how often did the taker actually win?
    Returns win_rate and sample_size.
    """
    try:
        result = _con().execute(f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM read_parquet('{_MARKETS}')
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND ticker LIKE '{category_pattern}'
            )
            SELECT
                AVG(CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END) AS win_rate,
                COUNT(*) AS sample_size
            FROM read_parquet('{_TRADES}') t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.yes_price = {price}
        """).fetchone()

        if result and result[1] > 0:
            return {"win_rate": round(result[0], 4), "sample_size": int(result[1])}
        return {"win_rate": None, "sample_size": 0}
    except Exception as e:
        return {"win_rate": None, "sample_size": 0, "error": str(e)}


def get_longshot_bias_stats(price_ceiling: int = 20, category_pattern: str = "KXNBA%") -> dict:
    """
    For all trades at or below price_ceiling (longshot YES side),
    how often did NO actually win? This is the core longshot bias metric.
    A high no_win_rate = strong bias = real edge fading the longshot.
    """
    try:
        result = _con().execute(f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM read_parquet('{_MARKETS}')
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND ticker LIKE '{category_pattern}'
            )
            SELECT
                AVG(CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END) AS no_win_rate,
                AVG(t.yes_price) AS avg_price,
                COUNT(*) AS sample_size
            FROM read_parquet('{_TRADES}') t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.yes_price <= {price_ceiling}
        """).fetchone()

        if result and result[2] > 0:
            return {
                "no_win_rate":  round(result[0], 4),
                "avg_price":    round(result[1], 2),
                "sample_size":  int(result[2]),
            }
        return {"no_win_rate": None, "avg_price": None, "sample_size": 0}
    except Exception as e:
        return {"no_win_rate": None, "sample_size": 0, "error": str(e)}


def get_price_bucket_edge(price: int, action: str, category_pattern: str = "KXNBA%") -> dict:
    """
    For a specific price and intended action (BET_NO or BET_YES),
    calculates the estimated edge = actual_win_rate - implied_probability.
    Positive edge = historically profitable at this price.
    """
    try:
        bet_side = "no" if action == "BET_NO" else "yes"
        implied_prob = (100 - price) / 100.0 if action == "BET_NO" else price / 100.0

        result = _con().execute(f"""
            WITH resolved AS (
                SELECT ticker, result
                FROM read_parquet('{_MARKETS}')
                WHERE status = 'finalized'
                  AND result IN ('yes', 'no')
                  AND ticker LIKE '{category_pattern}'
            )
            SELECT
                AVG(CASE WHEN m.result = '{bet_side}' THEN 1.0 ELSE 0.0 END) AS actual_win_rate,
                COUNT(*) AS sample_size
            FROM read_parquet('{_TRADES}') t
            INNER JOIN resolved m ON t.ticker = m.ticker
            WHERE t.yes_price = {price}
        """).fetchone()

        if result and result[1] > 0:
            actual_win_rate = round(result[0], 4)
            edge = round(actual_win_rate - implied_prob, 4)
            return {
                "actual_win_rate": actual_win_rate,
                "implied_prob":    round(implied_prob, 4),
                "edge":            edge,
                "sample_size":     int(result[1]),
            }
        return {"actual_win_rate": None, "implied_prob": round(implied_prob, 4), "edge": None, "sample_size": 0}
    except Exception as e:
        return {"edge": None, "sample_size": 0, "error": str(e)}


def get_market_volume_stats(ticker: str) -> dict:
    """
    Returns volume and open interest for a specific ticker.
    Used to filter out illiquid markets.
    """
    try:
        result = _con().execute(f"""
            SELECT volume, volume_24h, open_interest, last_price
            FROM read_parquet('{_MARKETS}')
            WHERE ticker = '{ticker}'
            LIMIT 1
        """).fetchone()

        if result:
            return {
                "volume":        int(result[0]) if result[0] else 0,
                "volume_24h":    int(result[1]) if result[1] else 0,
                "open_interest": int(result[2]) if result[2] else 0,
                "last_price":    int(result[3]) if result[3] else 0,
            }
        return {"volume": 0, "volume_24h": 0, "open_interest": 0, "last_price": 0}
    except Exception as e:
        return {"volume": 0, "open_interest": 0, "error": str(e)}