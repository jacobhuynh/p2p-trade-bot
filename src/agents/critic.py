"""
src/agents/critic.py

The Behavioral Critic — Bias Hunter / Adversarial Agent.

Its sole job is to find reasons NOT to take a trade that the orchestrator
marked as READY. It specifically hunts for:
  - YES/NO asymmetry exploitation being incorrectly applied
  - Sample size issues masking false edges
  - Market type mismatches (season wins vs game winner vs player prop)
  - Suspicious data patterns (win rate too perfect = data contamination)
  - Kelly fraction sanity checks
  - Recency concerns
  - Liquidity traps
  - Portfolio concentration (same game/team correlated exposure)

Before calling the LLM, the critic queries SQLite for all PENDING_RESOLUTION
trades to build an open portfolio snapshot.  Correlation is detected by
comparing the game identifier embedded in the ticker.

Only READY decisions get sent to the critic.
Output: APPROVE or VETO with a specific reason.
"""

import json
import re

_SYSTEM_PROMPT = """You are an adversarial trading critic for a Kalshi prediction market bot.
Your ONLY job is to find legitimate reasons to VETO a trade.

You are the last line of defense before real money is placed.
You are naturally skeptical. You assume the edge might be fake until proven otherwise.

CRITICAL CONTEXT — Favorite-Longshot Bias (FLB):
In prediction markets, underdogs (longshots) are systematically OVERPRICED relative to
their true win probability. This means favorites are systematically UNDERPRICED.
Our strategy CORRECTLY exploits this structural inefficiency:
  - BET_NO on heavy longshots (YES ≤ 20¢): fading the optimism/longshot tax
  - BET_YES on heavy favorites (YES ≥ 80¢): buying the underpriced favorite

NEVER VETO a trade because of "poor risk/reward ratio" or "risking X to win Y" if
calibration_gap > 0. Asymmetric payoff profiles (e.g., risking 88¢ to win 12¢) are
the EXPECTED structure of FLB trades — not a flaw. Only veto if the data is bad.

The following are STRICTLY FORBIDDEN as veto reasons:
  ✗ "low absolute dollar edge per contract"
  ✗ "poor risk/reward ratio" or "asymmetric risk"
  ✗ "risking X cents to win only Y cents"
  ✗ Lack of current-season team performance data or recent form
  ✗ Lack of matchup-specific data for this particular team pairing
Our edge comes from AGGREGATE PRICE BUCKET behavior across thousands of historical
contracts — NOT from predicting this specific game or team matchup. Team stats and
recent form are supplemental context only; their absence is never a veto reason.

You will receive:
- The original trade signal (ticker, price, action)
- The orchestrator's decision (READY with confidence/edge/kelly)
- The quant's full report (historical data, calibration gap, sample size)
- The current open portfolio (all PENDING_RESOLUTION trades)

You are specifically hunting for these failure modes:

1. YES/NO ASYMMETRY MISAPPLICATION
   - YES longshots (price <= 20c) are overpriced by up to 64pp vs NO longshots
   - If we're BET_NO on a YES longshot, that's correct — fading the optimism tax
   - If we're BET_YES on a NO longshot (price >= 80c), verify the asymmetry still holds
   - Flag if the yes_no_asymmetry value doesn't support the direction of the trade

2. SUSPICIOUS DATA PATTERNS
   - actual_win_rate of exactly 1.0 or 0.0 = almost certainly data contamination
   - calibration_gap > 0.50 = implausibly large, likely bad data
   - sample_size < 100 with EDGE_CONFIRMED verdict = overfitted

   IMPORTANT — MULTIPLE WIN RATES ARE DIFFERENT QUERIES, NOT COMPLEMENTS:
   The quant report contains several win-rate figures from separate DuckDB queries:
     • price_bucket_edge.actual_win_rate — how often our bet side won at this exact price
     • longshot_bias.no_win_rate — aggregate NO win rate across all YES longshots ≤ price
     • taker_win_rate.win_rate — raw taker win rate regardless of bet side
   These are computed from DIFFERENT population cuts and filters. They do NOT need to
   sum to 1.0 or complement each other. Both can legitimately be > 0.5.
   DO NOT flag this as contamination. Trust the pre-computed calibration_gap.

3. MARKET TYPE MISMATCH
   - Season win total markets (KXNBAWINS) behave differently than game winners (KXNBAGAME)
   - Player props (KXNBASGPROP) have different bias profiles than game outcomes
   - Flag if the historical data was aggregated across mixed market types

4. KELLY FRACTION CONCERNS
   - kelly_fraction > 0.10 on MEDIUM confidence = overbetting
   - kelly_fraction > 0.15 on any confidence = always flag

5. RECENCY / REGIME CHANGE
   - If sample spans multiple seasons, early seasons may not reflect current NBA dynamics
   - Flag if we can't confirm the edge holds in recent data

6. LIQUIDITY TRAP
   - live_open_interest < 100 is already pre-blocked in the bouncer (you won't see those)
   - orderbook_depth_at_price == 0 is pre-blocked by a hard Python rule (you won't see those)
   - If live_open_interest < 500 (check quant report), flag as a concern
   - If orderbook_depth_at_price is small relative to our intended contracts, flag as concern
   - orderbook_depth_at_price = None means we're offline — skip this check

7. MISSING SUPPLEMENTAL DATA
   - If game_context (ESPN) or team_stats (nba_api) is null/unavailable, this is
     normal — APIs can be slow or temporarily unavailable. Do NOT use missing
     supplemental data as a reason to VETO.
   - If sample_size >= 1000, APPROVE regardless of missing supplemental context.
     Large sample sizes make supplemental data unnecessary for confidence.

8. PORTFOLIO CONCENTRATION
   - Hard portfolio rules have ALREADY been enforced before you were called:
     opposing same-game positions, duplicate same-direction bets, and same-game
     dollar exposure ≥ $15 are all pre-blocked in Python. You will never see a
     trade that violates those rules.
   - If total open portfolio exposure > $50, flag overall concentration as a concern
   - All positions on the same game resolve together — note correlated loss risk

9. SENTIMENT / LIVE CONTEXT — ALWAYS ADDRESS
   - You receive a "Synthesized report" (quant + sentiment) and "Sentiment context" (live ESPN/news).
   - You MUST always consider this sentiment/live context (or note that it was absent or unavailable).
   - In your response you MUST populate the sentiment_note field with exactly one sentence stating how
     sentiment factored into your decision — e.g. aligned with the edge, conflicted but not decisive,
     absent/unavailable, or not weighted because the quantitative sample was large enough.
   - This is required for every response regardless of APPROVE/VETO and regardless of how much weight
     you gave to sentiment. The sentence can be brief (e.g. "Sentiment was unavailable; decision based on quant only.").

PRICE BUCKET MODEL — HOW OUR EDGE WORKS:
Our quantitative edge is derived from historical aggregate behavior of ALL NBA contracts
at a given yes_price, not from predicting individual team matchups. The sample_size in
the quant report reflects how many times contracts at this exact price were traded and
resolved — across all NBA teams, all seasons.

Therefore:
  - "We don't have data on this specific team's current season" = INVALID veto reason
  - "The historical sample mixes different team matchups" = INVALID veto reason (by design)
  - "Recent form is unknown" = INVALID veto reason (supplemental context only)
The only valid data-quality veto is sample_size < 100 or calibration_gap being None/implausible.

DATA FORMATTING: Ignore minor formatting discrepancies (e.g., timestamp "20260225"
vs "2026-02-25", or float vs int representations of the same number). These are
display differences, not data quality issues.

SMALL KELLY FRACTION: If kelly_fraction < 0.02 (2%), this is an exploratory micro-bet
with minimal capital at risk. On marginal cases where you're unsure, lean toward APPROVE
rather than VETO — the downside is capped by the small position size.

Respond ONLY with a JSON object — no extra text:
{
    "decision":      "APPROVE" or "VETO",
    "veto_reason":   "<specific reason if VETO, null if APPROVE>",
    "concerns":      ["<list of concerns even if approving — can be empty>"],
    "risk_score":    <int 1-10, where 10 = maximum risk>,
    "summary":       "<one sentence overall assessment>",
    "sentiment_note": "<one sentence: how sentiment/live context factored in, or that it was absent or not weighted>"
}

Be specific in veto_reason. "Insufficient data" is not specific enough.
"actual_win_rate of 1.0 across 7850 samples suggests data contamination,
not genuine edge" is specific enough.

You should APPROVE more often than you VETO — only hard block on clear issues.
"""


def _parse_game_key(ticker: str) -> str:
    """
    Extract the game identifier from a KXNBAGAME ticker for correlation detection.

    KXNBAGAME-25JAN15LACBOS-NO  →  "25JAN15LACBOS"

    For non-KXNBAGAME tickers, returns the ticker itself so at least exact
    duplicates are caught.
    """
    parts = ticker.split("-")
    return parts[1] if len(parts) >= 3 else ticker


def _check_hard_rules(
    action: str,
    same_game_trades: list[dict],
    same_game_exposure: float,
    quant_report: dict,
) -> dict | None:
    """
    Hard portfolio-safety rules enforced in Python before the LLM is called.
    Returns a fully-formed VETOED dict if any rule is violated, else None.

    Rules (in priority order):
      A. No opposing same-game positions (locks in guaranteed losses)
      B. No duplicate same-direction bets on the same game (over-concentration)
      C. Hard same-game dollar cap at $15
      D. Confirmed zero order book depth at our target price (online-only)
    """
    opposing_action = "BET_NO" if action == "BET_YES" else "BET_YES"

    # Rule A: opposing same-game position
    opposing = [t for t in same_game_trades if t.get("action") == opposing_action]
    if opposing:
        reason = (
            f"Opposing same-game position already open: holding {opposing_action} on "
            f"this game. Adding {action} would lock in guaranteed losses on one leg."
        )
        return _hard_veto(reason)

    # Rule B: duplicate same-direction same-game bet
    same_dir = [t for t in same_game_trades if t.get("action") == action]
    if same_dir:
        reason = (
            f"Duplicate same-game same-direction bet: already have "
            f"{len(same_dir)} open {action} trade(s) on this game. "
            f"Max 1 position per game per direction."
        )
        return _hard_veto(reason)

    # Rule C: same-game dollar cap
    if same_game_exposure >= 15.0:
        reason = (
            f"Same-game exposure ${same_game_exposure:.2f} already at or above "
            f"the $15 hard limit. No additional same-game exposure allowed."
        )
        return _hard_veto(reason)

    # Rule D: confirmed zero order book depth (None = offline/unknown → skip)
    depth = quant_report.get("orderbook_depth_at_price")
    if depth is not None and depth == 0:
        return _hard_veto(
            "Live order book confirms 0 contracts available at this price — trade cannot be filled"
        )

    return None


def _hard_veto(reason: str) -> dict:
    """Construct a VETOED decision dict for a hard-rule violation (no LLM call)."""
    return {
        "status":       "VETOED",
        "action":       "PASS",
        "critic": {
            "decision":       "VETO",
            "veto_reason":    reason,
            "concerns":       ["[Hard rule — no LLM call made]"],
            "risk_score":     10,
            "summary":        reason,
            "sentiment_note": "Not evaluated — trade blocked by hard portfolio rule before LLM.",
        },
    }


class CriticAgent:
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=0.2,  # Slightly higher than 0 — we want some creative skepticism
        )

    def review(self, trade_packet: dict, orchestrator_decision: dict) -> dict:
        """
        Reviews an orchestrator READY decision and either APPROVEs or VETOs it.
        Only call this when decision['status'] == 'READY'.

        Queries the open portfolio from SQLite before calling the LLM so the
        critic can check for correlated exposure on the same game.

        Returns the final decision dict with critic fields added.
        """
        ticker       = trade_packet.get("ticker", "")
        price        = trade_packet.get("market_price")
        action       = orchestrator_decision.get("action")
        quant        = orchestrator_decision.get("quant_summary", {})
        confidence   = orchestrator_decision.get("confidence")
        edge         = orchestrator_decision.get("edge")
        kelly        = orchestrator_decision.get("kelly_fraction")

        # ── Query open portfolio ───────────────────────────────────────────────
        try:
            from src.execution.trade_logger import TradeLogger
            open_trades = TradeLogger().open_trades()
        except Exception:
            open_trades = []

        game_key          = _parse_game_key(ticker)
        same_game_trades  = [t for t in open_trades if _parse_game_key(t["ticker"]) == game_key]
        total_exposure    = round(sum(t["cost_usd"] for t in open_trades), 2)
        same_game_exposure = round(sum(t["cost_usd"] for t in same_game_trades), 2)

        # ── Hard rules — block before wasting an LLM call ─────────────────────
        hard_veto = _check_hard_rules(action, same_game_trades, same_game_exposure, quant)
        if hard_veto is not None:
            return {**orchestrator_decision, **hard_veto}

        portfolio_section = (
            f"Open Portfolio ({len(open_trades)} PENDING_RESOLUTION trade(s), "
            f"${total_exposure:.2f} total exposure):\n"
            f"  Same-game trades : {len(same_game_trades)} trade(s), "
            f"${same_game_exposure:.2f} exposure on this game"
        )

        # Detect market type from ticker prefix
        if "KXNBAWINS" in ticker:
            market_type_context = "SEASON WIN TOTAL — different bias profile than game markets"
        elif "KXNBASGPROP" in ticker:
            market_type_context = "PLAYER PROP — different bias profile than game markets"
        else:
            market_type_context = "GAME WINNER — standard longshot bias applies"

        sentiment_context = trade_packet.get("sentiment_context") or "None."
        synthesized_report = orchestrator_decision.get("synthesized_report") or "None."
        human_msg = f"""Trade to review:
Ticker:         {ticker}
Market Type:    {market_type_context}
Price:          {price}c
Action:         {action}
Confidence:     {confidence}
Edge:           {edge}
Kelly Fraction: {kelly}

{portfolio_section}

Synthesized report (Quant + Sentiment):
{synthesized_report}

Sentiment context (live ESPN/news):
{sentiment_context}

Quant Report:
{json.dumps(quant, indent=2)}

Orchestrator Decision:
{json.dumps(orchestrator_decision, indent=2)}

Find reasons to VETO this trade. Be specific.
"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = self.llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw    = response.content
            clean  = re.sub(r"```json|```", "", raw).strip()
            critique = json.loads(clean)

        except Exception as e:
            critique = {
                "decision":       "VETO",
                "veto_reason":    f"Critic agent error — defaulting to VETO for safety: {str(e)}",
                "concerns":       [],
                "risk_score":     10,
                "summary":        "Critic failed to run — trade blocked for safety.",
                "sentiment_note": "Not specified.",
            }

        # Merge critic result into the decision object
        final_status = "APPROVED" if critique["decision"] == "APPROVE" else "VETOED"

        return {
            **orchestrator_decision,
            "status":       final_status,
            "action":       orchestrator_decision["action"] if final_status == "APPROVED" else "PASS",
            "critic": {
                "decision":       critique["decision"],
                "veto_reason":   critique.get("veto_reason"),
                "concerns":      critique.get("concerns", []),
                "risk_score":    critique.get("risk_score"),
                "summary":       critique.get("summary"),
                "sentiment_note": critique.get("sentiment_note") or "Not specified.",
            }
        }
