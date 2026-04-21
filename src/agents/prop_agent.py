"""
src/agents/prop_agent.py

PlayerPropAgent — Stats-based edge analyzer for KXNBASGPROP markets.

Edge is computed purely from nba_api data: compare the Kalshi prop line to
the player's recent average and variance.  No historical parquet calibration
data exists for player props (unlike game winners), so we rely on recent
performance trends.

All math is computed in Python before the LLM call.  The LLM's only job is
to write a single-sentence qualitative summary.

Kelly sizing is capped at 0.05 (vs. 0.15 for game winners) to reflect the
higher variance inherent in individual player stat lines.
"""

import json
import re
import statistics
from typing import Optional

_KELLY_CAP = 0.05

_SYSTEM_PROMPT = """You are a sports betting analyst for a Kalshi prediction market trading bot.

All numerical calculations have already been computed in Python and are provided as verified facts.
Do NOT recompute them.

Your task is to write a 3-sentence qualitative summary:
  Sentence 1 — Trend: State the hit rate, recent average vs. the prop threshold, and the verdict.
  Sentence 2 — Context: Note any matchup history vs. the opponent and usage/pace factors if available.
  Sentence 3 — Risk: Flag concerns like small sample size, usage volatility, injury risk, or why the edge might not hold.

Respond ONLY with a JSON object — no extra text:
{"summary": "<three sentences>"}
"""


class PlayerPropAgent:
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
        )

    def analyze(self, trade_packet: dict) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        player_name    = trade_packet.get("player_name", "")
        prop_type      = trade_packet.get("prop_type", "")      # PTS | REB | AST
        prop_threshold = trade_packet.get("prop_threshold")      # numeric threshold
        action         = trade_packet.get("action", "")          # BET_YES | BET_NO
        yes_price      = trade_packet.get("market_price", 50)
        opponent_abbr  = trade_packet.get("opponent_abbr", "")   # may be absent

        # ── Fetch player data (all graceful — never block) ─────────────────────
        recent_stats:    Optional[dict] = None
        usage_data:      Optional[dict] = None
        matchup_history: Optional[dict] = None

        try:
            from src.tools.nba_player_stats_tool import get_player_recent_stats
            recent_stats = get_player_recent_stats(player_name)
        except Exception:
            pass

        try:
            from src.tools.nba_player_stats_tool import get_player_usage_rate
            usage_data = get_player_usage_rate(player_name)
        except Exception:
            pass

        if opponent_abbr:
            try:
                from src.tools.nba_player_stats_tool import get_player_matchup_history
                matchup_history = get_player_matchup_history(player_name, opponent_abbr)
            except Exception:
                pass

        # ── Compute edge in Python ─────────────────────────────────────────────
        recent_avg: Optional[float] = None
        hit_rate:   Optional[float] = None
        variance:   Optional[float] = None
        edge:       Optional[float] = None
        raw_values: list[float]     = []

        stat_key_map = {"PTS": "avg_pts", "REB": "avg_reb", "AST": "avg_ast"}
        game_stat_map = {"PTS": "pts", "REB": "reb", "AST": "ast"}

        if recent_stats and prop_threshold is not None:
            avg_key  = stat_key_map.get(prop_type)
            game_key = game_stat_map.get(prop_type)

            if avg_key and avg_key in recent_stats:
                recent_avg = recent_stats[avg_key]

            if game_key and recent_stats.get("games"):
                raw_values = [g[game_key] for g in recent_stats["games"] if game_key in g]

            if raw_values and prop_threshold is not None:
                exceeded   = [v for v in raw_values if v >= prop_threshold]
                hit_rate   = round(len(exceeded) / len(raw_values), 4)
                variance   = round(statistics.stdev(raw_values), 2) if len(raw_values) > 1 else None
                if recent_avg is not None:
                    edge = round(recent_avg - float(prop_threshold), 2)

        n_games = len(raw_values)

        # ── Verdict ────────────────────────────────────────────────────────────
        if n_games < 5 or hit_rate is None:
            verdict      = "INSUFFICIENT_DATA"
            data_quality = "INSUFFICIENT"
        elif hit_rate > 0.65:
            verdict      = "EDGE_CONFIRMED"
            data_quality = "SUFFICIENT"
        elif hit_rate > 0.55:
            verdict      = "EDGE_WEAK"
            data_quality = "SUFFICIENT"
        else:
            verdict      = "NO_EDGE"
            data_quality = "SUFFICIENT"

        # ── Kelly sizing (conservative for props) ──────────────────────────────
        # Scale kelly by how far hit_rate exceeds 0.5 (breakeven).
        # Cap tightly at _KELLY_CAP to reflect higher per-game variance.
        if hit_rate is not None and hit_rate > 0.5:
            raw_kelly = (hit_rate - 0.5) / (1 - hit_rate + 1e-9)
            kelly_fraction = round(min(raw_kelly, _KELLY_CAP), 4)
        else:
            kelly_fraction = 0.01

        confidence = (
            "HIGH"   if verdict == "EDGE_CONFIRMED" else
            "MEDIUM" if verdict == "EDGE_WEAK"      else
            "LOW"
        )

        # ── LLM summary ────────────────────────────────────────────────────────
        human_msg = f"""Pre-computed Player Prop Analysis:
Player:          {player_name}
Prop Type:       {prop_type}
Prop Threshold:  {prop_threshold}
Action:          {action} (yes_price={yes_price}c)
Recent Avg:      {recent_avg}  ({n_games} games sampled)
Hit Rate:        {hit_rate}  (fraction of last {n_games} games exceeding threshold)
Variance (std):  {variance}
Edge:            {edge}  (recent_avg - threshold)
Verdict:         {verdict}
Data Quality:    {data_quality}

Usage Rate:
{json.dumps(usage_data, indent=2) if usage_data else "unavailable"}

Matchup History vs {opponent_abbr or "N/A"}:
{json.dumps(matchup_history, indent=2) if matchup_history else "unavailable or no opponent context"}

Write a single-sentence qualitative summary. Use the pre-computed values — do not recalculate.
"""

        summary = ""
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
            summary = f"Prop summary unavailable: {str(e)}"

        return {
            "player_name":     player_name,
            "prop_type":       prop_type,
            "prop_threshold":  prop_threshold,
            "action":          action,
            "price":           yes_price,
            "side":            "yes" if action == "BET_YES" else "no",
            "ticker":          trade_packet.get("ticker", ""),
            "recent_avg":      recent_avg,
            "hit_rate":        hit_rate,
            "variance":        variance,
            "edge":            edge,
            "n_games_sampled": n_games,
            "verdict":         verdict,
            "data_quality":    data_quality,
            "confidence":      confidence,
            "kelly_fraction":  kelly_fraction,
            "summary":         summary,
            "matchup_context": matchup_history,
            "usage_rate":      usage_data,
            # Mirrors quant_summary shape for trade_logger compatibility
            "quant_summary": {
                "calibration_gap": edge,
                "actual_win_rate": hit_rate,
                "sample_size":     n_games,
                "verdict":         verdict,
            },
        }
