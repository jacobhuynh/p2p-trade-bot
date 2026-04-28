"""
src/web/mock_fixtures.py

Hand-crafted fixtures used by POST /api/mock to exercise the frontend
without waiting for a live Kalshi ticker.  Each fixture mirrors the shape
that LeadAnalyst.analyze_signal / analyze_prop_signal would produce, so
the workflow viz, trade detail drawer, and live feed all behave as if a
real decision had flowed through.
"""

from __future__ import annotations

import random
import string


def _suffix() -> str:
    return "".join(random.choices(string.digits, k=4))


def game_winner_fixture() -> tuple[dict, dict]:
    """Return (trade_packet, decision) for a GAME_WINNER mock decision."""
    suffix = _suffix()
    trade_packet = {
        "ticker": f"KXNBAGAME-MOCK{suffix}-LAL",
        "market_price": 14,
        "category": "NBA",
        "action": "BET_NO",
        "reason": "Longshot Bias - fading overpriced YES underdog at 14c",
        "market_title": "Los Angeles Lakers @ Boston Celtics — Lakers to win?",
        "market_type": "binary",
        "rules_primary": "Resolves YES if Los Angeles Lakers win the regulation game.",
        "live_open_interest": 8421,
        "live_volume_24h": 23877,
        "contract_type": "GAME_WINNER",
        "sentiment_context": (
            "ESPN: Lakers travel into Boston on the back end of a back-to-back.\n"
            "Davis questionable (knee). Tatum returns from one-game absence.\n"
            "Vegas opener has Boston -8.5; market consensus skews heavily Celtics."
        ),
    }
    decision = {
        "ticker": trade_packet["ticker"],
        "action": "BET_NO",
        "price": 14,
        "side": "no",
        "status": "APPROVED",
        "confidence": "HIGH",
        "edge": 0.0235,
        "kelly_fraction": 0.15,
        "reason": (
            "Strong calibration edge at 14c price bucket; longshot bias firmly negative; "
            "ESPN context favors Celtics; sentiment confirms travel + injury disadvantage."
        ),
        "quant_summary": {
            "calibration_gap": 0.0235,
            "actual_win_rate": 0.0935,
            "implied_prob": 0.14,
            "sample_size": 1842,
            "data_quality": "HIGH",
            "verdict": "EDGE_CONFIRMED",
            "no_win_rate": 0.9065,
            "yes_no_asymmetry": 0.0072,
            "price_bucket_edge": {"actual_win_rate": 0.092, "edge": 0.048, "sample_size": 612},
            "longshot_bias": {"no_win_rate": 0.91, "avg_price": 0.135, "sample_size": 612},
            "taker_win_rate": {"win_rate": 0.088, "sample_size": 433},
            "inverse_bucket": {"actual_win_rate": 0.905, "edge": 0.045, "sample_size": 598},
            "game_context": {
                "away_abbr": "LAL", "home_abbr": "BOS",
                "away_score": 0, "home_score": 0,
                "status": "STATUS_SCHEDULED", "winner_abbr": None,
            },
            "team_stats": {
                "home": {"abbr": "BOS", "last10": "8-2", "home_record": "21-5", "away_record": "16-9"},
                "away": {"abbr": "LAL", "last10": "5-5", "home_record": "14-12", "away_record": "11-15"},
            },
            "summary": (
                "Lakers priced at 14c on a back-to-back into TD Garden; historical bucket "
                "wins only ~9% of the time. Edge favours BET_NO."
            ),
        },
        "critic": {
            "decision": "APPROVE",
            "risk_score": 3,
            "veto_reason": None,
            "concerns": [
                "Kelly fraction at 0.15 cap — limited upside scaling room.",
                "AD injury status could move the line meaningfully before tip.",
            ],
            "summary": "Edge is real and well-supported by both quant and sentiment.",
            "sentiment_note": "ESPN context aligns with quant verdict; no contradicting news.",
        },
    }
    return trade_packet, decision


def player_prop_fixture() -> tuple[dict, dict]:
    """Return (trade_packet, decision) for a PLAYER_PROP mock decision."""
    suffix = _suffix()
    trade_packet = {
        "ticker": f"KXNBAPTS-MOCK{suffix}-JOKIC",
        "market_price": 17,
        "action": "BET_NO",
        "market_title": "Will Nikola Jokic record 12+ assists?",
        "contract_type": "PLAYER_PROP",
        "player_name": "Nikola Jokic",
        "prop_type": "AST",
        "prop_threshold": 12.0,
        "live_open_interest": 1342,
        "sentiment_context": (
            "Jokic posted only 6 and 8 assists in his last two games; Murray returning "
            "from one-game absence which historically reduces Jokic's playmaking load."
        ),
    }
    decision = {
        "ticker": trade_packet["ticker"],
        "action": "BET_NO",
        "price": 17,
        "side": "no",
        "status": "APPROVED",
        "confidence": "HIGH",
        "edge": 0.68,
        "kelly_fraction": 0.05,
        "reason": (
            "Recent AST trend below line; matchup history favors UNDER; Murray return "
            "reduces Jokic's projected assist share."
        ),
        "player_name": "Nikola Jokic",
        "prop_type": "AST",
        "prop_threshold": 12.0,
        "quant_summary": {
            "prop_type": "AST",
            "prop_threshold": 12.0,
            "actual_win_rate": 0.68,
            "hit_rate": 0.32,
            "effective_win_rate": 0.68,
            "recent_avg": 9.4,
            "calibration_gap": -2.6,
            "variance": 8.9,
            "sample_size": 28,
            "n_games_sampled": 28,
            "data_quality": "SUFFICIENT",
            "verdict": "EDGE_CONFIRMED",
            "matchup_context": {
                "avg_pts_vs_opp": 27.5, "avg_reb_vs_opp": 12.1,
                "avg_ast_vs_opp": 8.6, "n_games": 6,
            },
            "summary": (
                "Jokic averaging 9.4 AST over last 28 with 32% hit rate at 12+; matchup "
                "history vs opponent shows 8.6 AST avg over 6 games."
            ),
        },
        "critic": {
            "decision": "APPROVE",
            "risk_score": 4,
            "veto_reason": None,
            "concerns": [
                "Sample size 28 games — borderline trustworthy.",
                "Variance high (σ²=8.9) — single hot game can flip outcome.",
            ],
            "summary": "Edge is real but modest; sized appropriately at 9% Kelly.",
            "sentiment_note": "Murray return is a meaningful tailwind for the BET_NO thesis.",
        },
    }
    return trade_packet, decision
