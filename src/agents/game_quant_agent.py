"""
src/agents/game_quant_agent.py

The Quant Agent — Historical Edge / Calibration Gap Analyzer for GAME_WINNER markets.

Key insight: we never query by ticker (live tickers won't exist in historical DB).
Instead we query by PRICE BUCKET across all finalized NBA markets.
The question is: "Historically, what actually happened to NBA contracts priced at X cents?"

All math (calibration gap, verdict, implied probability) is computed in Python
BEFORE calling the LLM.  The LLM's only job is to write a qualitative one-sentence
summary that incorporates the pre-computed numbers and any available live context
(ESPN game status, recent team records from nba_api).

Real prediction market edges are small:
  - 1.5%+ calibration gap = strong edge
  - 0.75%+ calibration gap = weak but real
  - <0.75% = noise, not tradeable
"""

import json
import re

from src.tools.duckdb_tool import (
    get_historical_win_rate,
    get_longshot_bias_stats,
    get_price_bucket_edge,
    get_market_volume_stats,
)



_SYSTEM_PROMPT = """You are a quantitative analyst for a Kalshi prediction market trading bot.

All numerical calculations (calibration gap, implied probability, verdict) have already
been computed in Python and are provided to you as verified facts.  Do NOT recompute them.

Your task is to write a 3-sentence qualitative summary:
  Sentence 1 — Edge: State the calibration gap, verdict, and implied vs. actual win rate.
  Sentence 2 — Context: Note team momentum (recent records), live game status, and any key player trends (hot/cold streaks, scoring leaders) if available.
  Sentence 3 — Risk: Flag data quality concerns (low sample, perfect win rate, low liquidity) or reasons this edge might not hold.

Respond ONLY with a JSON object — no extra text:
{"summary": "<three sentences>"}
"""


class QuantAgent:
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
        )

    def analyze(self, trade_packet: dict) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        price  = trade_packet.get("market_price")
        action = trade_packet.get("action")
        ticker = trade_packet.get("ticker", "")

        # ── Query by PRICE BUCKET, not ticker ─────────────────────────────────
        edge_data    = get_price_bucket_edge(price, action)
        bias_data    = get_longshot_bias_stats(price)
        win_data     = get_historical_win_rate(price)
        inverse_edge = get_price_bucket_edge(100 - price, "BET_YES" if action == "BET_NO" else "BET_NO")

        # ── Compute all math in Python ─────────────────────────────────────────
        implied_prob    = round((100 - price) / 100, 4) if action == "BET_NO" else round(price / 100, 4)
        calibration_gap = edge_data.get("edge")           # already Python-computed by duckdb_tool
        actual_win_rate = edge_data.get("actual_win_rate")
        sample_size     = edge_data.get("sample_size", 0)

        # yes_no_asymmetry: difference between our side edge and the inverse side edge
        inv_edge_val = inverse_edge.get("edge")
        yes_no_asymmetry = (
            round(calibration_gap - inv_edge_val, 4)
            if calibration_gap is not None and inv_edge_val is not None
            else None
        )

        # Verdict (mirrors previous LLM rules, now enforced in Python)
        if sample_size < 100 or calibration_gap is None:
            verdict      = "INSUFFICIENT_DATA"
            data_quality = "INSUFFICIENT"
        elif calibration_gap > 0.015 and sample_size >= 200:
            verdict      = "EDGE_CONFIRMED"
            data_quality = "SUFFICIENT"
        elif calibration_gap > 0.0075 and sample_size >= 100:
            verdict      = "EDGE_WEAK"
            data_quality = "SUFFICIENT"
        else:
            verdict      = "NO_EDGE"
            data_quality = "SUFFICIENT"

        # ── Fetch live context (graceful — never blocks the pipeline) ──────────
        game_context: dict | None = None
        team_stats:   dict | None = None
        home_key_players: list | None = None
        away_key_players: list | None = None
        try:
            from src.tools.espn_tool import find_game
            game_context = find_game(ticker)
        except Exception:
            pass

        try:
            from src.tools.nba_team_tool import get_team_recent_records, _parse_teams_from_ticker
            team_stats = get_team_recent_records(ticker)
            parsed = _parse_teams_from_ticker(ticker)
            if parsed:
                from src.tools.nba_player_stats_tool import get_team_key_players
                home_abbr, away_abbr = parsed
                home_key_players = get_team_key_players(home_abbr)
                away_key_players = get_team_key_players(away_abbr)
        except Exception:
            pass

        # Orderbook depth check removed — snapshot depth is unreliable (resting orders
        # at a price can be 0 immediately after a trade even with active liquidity).
        # orderbook_depth_at_price stays None (unknown) so the critic skips the check.

        # ── Ask LLM for qualitative summary only ──────────────────────────────
        def _fmt_key_players(players: list | None, abbr: str) -> str:
            if not players:
                return f"{abbr}: unavailable"
            lines = [f"{abbr} key players:"]
            for p in players:
                l5 = ", ".join(str(x) for x in p.get("last5_pts", []))
                lines.append(
                    f"  {p['name']}: {p['avg_pts']}pts/{p['avg_reb']}reb/{p['avg_ast']}ast avg  last5_pts=[{l5}]"
                )
            return "\n".join(lines)

        home_abbr_str = (team_stats or {}).get("home", {}).get("abbr", "HOME") if team_stats else "HOME"
        away_abbr_str = (team_stats or {}).get("away", {}).get("abbr", "AWAY") if team_stats else "AWAY"

        human_msg = f"""Pre-computed Analysis:
Ticker:          {ticker}
Price:           {price}c
Action:          {action}
Implied Prob:    {implied_prob} ({implied_prob*100:.1f}%)
Calibration Gap: {calibration_gap}  (actual_win_rate - implied_prob)
Actual Win Rate: {actual_win_rate}
Sample Size:     {sample_size}
Verdict:         {verdict}
Data Quality:    {data_quality}
Yes/No Asymmetry:{yes_no_asymmetry}
No Win Rate:     {bias_data.get('no_win_rate')} (longshot bias stat)

Live ESPN Context:
{json.dumps(game_context, indent=2) if game_context else "No game found for today/yesterday."}

Team Recent Records (nba_api):
{json.dumps(team_stats, indent=2) if team_stats else "unavailable"}

Key Player Stats — last 10 games average (nba_api):
{_fmt_key_players(home_key_players, home_abbr_str)}
{_fmt_key_players(away_key_players, away_abbr_str)}

Write a 3-sentence qualitative summary.  Use the pre-computed values — do not recalculate.
"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw   = response.content
            clean = re.sub(r"```json|```", "", raw).strip()
            llm_out = json.loads(clean)
            summary = llm_out.get("summary", "")
        except Exception as e:
            summary = f"Quant summary unavailable: {str(e)}"

        # ── Assemble and return — same keys as before so downstream is unchanged ─
        return {
            "historical_edge":          calibration_gap,
            "actual_win_rate":          actual_win_rate,
            "implied_prob":             implied_prob,
            "no_win_rate":              bias_data.get("no_win_rate"),
            "yes_no_asymmetry":         yes_no_asymmetry,
            "sample_size":              sample_size,
            "data_quality":             data_quality,
            "verdict":                  verdict,
            "calibration_gap":          calibration_gap,
            "summary":                  summary,
            "game_context":             game_context,
            "team_stats":               team_stats,
            "home_key_players":         home_key_players,
            "away_key_players":         away_key_players,
            # ── Raw query results — separate DuckDB queries, different population cuts ──
            "price_bucket_edge":        edge_data,    # get_price_bucket_edge(price, action)
            "longshot_bias":            bias_data,    # get_longshot_bias_stats(price)
            "taker_win_rate":           win_data,     # get_historical_win_rate(price)
            "inverse_bucket":           inverse_edge, # get_price_bucket_edge(100-price, opposite)
        }
