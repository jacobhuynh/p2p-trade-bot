"""
src/tools/nba_player_stats_tool.py

nba_api wrapper for player-level statistics (used by PropAgent and GameQuantAgent).

Fetches recent game logs, usage rate, and matchup history for a given player.
All calls are wrapped with graceful error handling — returns None on any failure
so the pipeline never blocks.

Public API
----------
get_player_recent_stats(player_name, last_n=10) -> dict | None
get_player_usage_rate(player_name) -> dict | None
get_player_matchup_history(player_name, opponent_abbr, last_n=5) -> dict | None
"""

import re
from typing import Optional

_NBA_API_TIMEOUT = 15


# ── Player lookup ──────────────────────────────────────────────────────────────

def _find_player_id(player_name: str) -> Optional[int]:
    """
    Return nba_api player ID for the given full name.
    Tries exact match first, then falls back to a case-insensitive partial match.
    """
    try:
        from nba_api.stats.static import players as nba_players

        name_clean = player_name.strip()
        results = nba_players.find_players_by_full_name(name_clean)
        if results:
            return results[0]["id"]

        # Fuzzy fallback: split into first/last and search individually
        parts = name_clean.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            candidates = nba_players.find_players_by_last_name(last_name)
            name_lower = name_clean.lower()
            for p in candidates:
                if p["full_name"].lower() == name_lower:
                    return p["id"]
            if candidates:
                return candidates[0]["id"]

        return None
    except Exception:
        return None


# ── Recent game log ────────────────────────────────────────────────────────────

def get_player_recent_stats(player_name: str, last_n: int = 10) -> Optional[dict]:
    """
    Fetch the last `last_n` games for a player and compute per-game averages.

    Returns:
        {
          "avg_pts": 27.3, "avg_reb": 7.1, "avg_ast": 6.4, "avg_min": 35.2,
          "games": [{"date": "2025-04-15", "pts": 30, "reb": 8, "ast": 5, "opp": "BOS"}, ...]
        }
        or None on failure.
    """
    player_id = _find_player_id(player_name)
    if player_id is None:
        return None

    try:
        from nba_api.stats.endpoints import PlayerGameLog

        log = PlayerGameLog(
            player_id=player_id,
            season="2024-25",
            timeout=_NBA_API_TIMEOUT,
        )
        df = log.get_data_frames()[0]
        if df is None or df.empty:
            return None

        recent = df.head(last_n)

        def _safe_avg(col: str) -> Optional[float]:
            if col in recent.columns:
                return round(float(recent[col].mean()), 2)
            return None

        games = []
        for _, row in recent.iterrows():
            matchup = str(row.get("MATCHUP", ""))
            opp = re.split(r"\s+vs\.\s+|\s+@\s+", matchup)[-1].strip() if matchup else ""
            games.append({
                "date": str(row.get("GAME_DATE", "")),
                "pts":  int(row.get("PTS", 0)),
                "reb":  int(row.get("REB", 0)),
                "ast":  int(row.get("AST", 0)),
                "opp":  opp,
            })

        return {
            "avg_pts": _safe_avg("PTS"),
            "avg_reb": _safe_avg("REB"),
            "avg_ast": _safe_avg("AST"),
            "avg_min": _safe_avg("MIN"),
            "games":   games,
        }
    except Exception:
        return None


# ── Usage rate ─────────────────────────────────────────────────────────────────

def get_player_usage_rate(player_name: str) -> Optional[dict]:
    """
    Return season-level usage rate, true shooting %, and pace context.

    Returns:
        {"usage_rate": 0.312, "ts_pct": 0.601, "pace": 100.4}
        or None on failure.
    """
    player_id = _find_player_id(player_name)
    if player_id is None:
        return None

    try:
        from nba_api.stats.endpoints import PlayerDashboardByGeneralSplits

        dash = PlayerDashboardByGeneralSplits(
            player_id=player_id,
            season="2024-25",
            timeout=_NBA_API_TIMEOUT,
        )
        df = dash.get_data_frames()[0]
        if df is None or df.empty:
            return None

        row = df.iloc[0]

        def _col(name: str):
            return float(row[name]) if name in df.columns else None

        return {
            "usage_rate": _col("USG_PCT"),
            "ts_pct":     _col("TS_PCT"),
            "pace":       _col("PACE"),
        }
    except Exception:
        return None


# ── Matchup history ────────────────────────────────────────────────────────────

def get_player_matchup_history(
    player_name: str,
    opponent_abbr: str,
    last_n: int = 5,
) -> Optional[dict]:
    """
    Return averages for a player's last `last_n` games against a specific opponent.

    Uses the full game log and filters rows where MATCHUP contains the opponent abbr.

    Returns:
        {
          "avg_pts_vs_opp": 29.0, "avg_reb_vs_opp": 6.0,
          "avg_ast_vs_opp": 7.5,  "n_games": 4
        }
        or None if no matching games or on failure.
    """
    player_id = _find_player_id(player_name)
    if player_id is None:
        return None

    try:
        from nba_api.stats.endpoints import PlayerGameLog

        log = PlayerGameLog(
            player_id=player_id,
            season="2024-25",
            timeout=_NBA_API_TIMEOUT,
        )
        df = log.get_data_frames()[0]
        if df is None or df.empty:
            return None

        opp_upper = opponent_abbr.upper()
        mask = df["MATCHUP"].str.contains(opp_upper, na=False)
        vs_opp = df[mask].head(last_n)

        if vs_opp.empty:
            return None

        def _avg(col: str) -> Optional[float]:
            if col in vs_opp.columns:
                return round(float(vs_opp[col].mean()), 2)
            return None

        return {
            "avg_pts_vs_opp": _avg("PTS"),
            "avg_reb_vs_opp": _avg("REB"),
            "avg_ast_vs_opp": _avg("AST"),
            "n_games":        len(vs_opp),
        }
    except Exception:
        return None


# ── Team key players ───────────────────────────────────────────────────────────

def get_team_key_players(team_abbr: str, top_n: int = 3) -> Optional[list[dict]]:
    """
    Return recent stats for the top `top_n` scorers on a team this season.

    Uses CommonTeamRoster to get the active roster, then fetches a short
    game log for each player and sorts by season scoring average.

    Returns a list of dicts, one per player:
        [
          {"name": "Jayson Tatum", "avg_pts": 26.4, "avg_reb": 8.1,
           "avg_ast": 4.9, "last5_pts": [28, 22, 31, 19, 25]},
          ...
        ]
        or None on failure.
    """
    try:
        from nba_api.stats.static import teams as nba_teams
        from nba_api.stats.endpoints import CommonTeamRoster, PlayerGameLog

        results = nba_teams.find_teams_by_abbreviation(team_abbr.upper())
        if not results:
            return None
        team_id = results[0]["id"]

        roster = CommonTeamRoster(team_id=team_id, timeout=_NBA_API_TIMEOUT)
        roster_df = roster.get_data_frames()[0]
        if roster_df is None or roster_df.empty:
            return None

        player_rows = roster_df[["PLAYER_ID", "PLAYER"]].dropna().to_dict("records")

        player_stats = []
        for p in player_rows:
            pid  = p["PLAYER_ID"]
            name = p["PLAYER"]
            try:
                log = PlayerGameLog(
                    player_id=pid,
                    season="2024-25",
                    timeout=_NBA_API_TIMEOUT,
                )
                df = log.get_data_frames()[0]
                if df is None or df.empty or len(df) < 3:
                    continue
                recent = df.head(10)
                avg_pts = round(float(recent["PTS"].mean()), 1)
                avg_reb = round(float(recent["REB"].mean()), 1)
                avg_ast = round(float(recent["AST"].mean()), 1)
                last5   = [int(v) for v in df.head(5)["PTS"].tolist()]
                player_stats.append({
                    "name":     name,
                    "avg_pts":  avg_pts,
                    "avg_reb":  avg_reb,
                    "avg_ast":  avg_ast,
                    "last5_pts": last5,
                })
            except Exception:
                continue

        if not player_stats:
            return None

        player_stats.sort(key=lambda x: x["avg_pts"], reverse=True)
        return player_stats[:top_n]

    except Exception:
        return None
