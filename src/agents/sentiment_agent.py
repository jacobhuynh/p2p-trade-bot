"""
src/agents/sentiment_agent.py

Sentiment Agent — fetches live ESPN game and team news for a matchup and
produces a short aggregated context for the trading workflow.

Runs only for GAME_WINNER contracts; for TOTALS / PLAYER_PROP returns the
packet unchanged (no-op). Uses claude-haiku-4-5 and a single ESPN tool
(get_espn_matchup_context) to fetch data, then summarizes into
trade_packet["sentiment_context"].
"""

import json

_SYSTEM_PROMPT = """You are a research assistant for a prediction market trading bot.
Your job is to use the provided tool to fetch live ESPN game status and team-relevant news for an NBA matchup, then write a short 2–4 sentence sentiment/live context summary for the trading workflow.

Include: game status if available, any injuries or lineup news, recent headlines about either team, and momentum or narrative that could affect the game. Be concise and factual.
Output ONLY the summary text — no preamble, no "Summary:" label, no JSON."""


def _espn_matchup_context_tool(ticker: str) -> str:
    """Fetch ESPN game status and team-relevant news for a KXNBAGAME ticker. Returns JSON string."""
    try:
        from src.tools.espn_tool import get_espn_matchup_context
        out = get_espn_matchup_context(ticker)
        return json.dumps(out) if out is not None else "{}"
    except Exception as e:
        return json.dumps({"error": str(e)})


class SentimentAgent:
    def __init__(self):
        from langchain_anthropic import ChatAnthropic
        from langchain_core.tools import tool

        # Tool the agent can call to access ESPN matchup data
        @tool
        def get_espn_matchup_context(ticker: str) -> str:
            """Fetch live ESPN game status and team-relevant news for an NBA game winner ticker (e.g. KXNBAGAME-25JAN15LACBOS-NO). Returns game info and recent headlines for the two teams."""
            return _espn_matchup_context_tool(ticker)

        self._tool = get_espn_matchup_context
        self.llm = ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
        ).bind_tools([get_espn_matchup_context])

    def enrich(self, trade_packet: dict) -> dict:
        """
        If contract_type is GAME_WINNER, fetch ESPN matchup context via tool,
        summarize with the LLM, and set trade_packet["sentiment_context"].
        Otherwise return the packet unchanged.
        """
        if trade_packet.get("contract_type") != "GAME_WINNER":
            return trade_packet

        ticker = trade_packet.get("ticker", "")
        if not ticker:
            return trade_packet

        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        prompt = (
            f"Ticker: {ticker}\n\n"
            "Use the get_espn_matchup_context tool to fetch live game and team news, "
            "then write a short 2–4 sentence sentiment/live context summary. "
            "Output only the summary text."
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = self.llm.invoke(messages)
        except Exception:
            trade_packet["sentiment_context"] = None
            return trade_packet

        # If LLM requested a tool call, run it and send result back for summary
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                if tc.get("name") == "get_espn_matchup_context":
                    args = tc.get("args", {})
                    ticker_arg = args.get("ticker", ticker)
                    result = _espn_matchup_context_tool(ticker_arg)
                    messages.append(response)
                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tc.get("id", ""),
                        )
                    )
                    try:
                        response = self.llm.invoke(messages)
                    except Exception:
                        trade_packet["sentiment_context"] = None
                        return trade_packet
                    break

        summary = (response.content or "").strip() if hasattr(response, "content") else ""
        trade_packet["sentiment_context"] = summary if summary else None
        return trade_packet
