"""
src/config.py

Central configuration for the p2p-trade-bot.
Reads all mode flags and trading parameters from the environment.
Supports .env files (via python-dotenv) but works fine without one.

Environment variables:
  LLM_MODE             rule (default) | anthropic
  EXECUTION_MODE       paper (default) | live | backtest
  PAPER_STARTING_CASH  starting cash in dollars (default 1000.0)
  PAPER_MAX_CONTRACTS  hard cap per trade (default 20)
  PAPER_DATA_DIR       where paper files live (default data/paper)
"""

import os
import warnings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to plain env vars


# ── LLM agent mode ───────────────────────────────────────────────────────────
_llm_raw = os.getenv("LLM_MODE", "rule").lower().strip()
if _llm_raw not in ("rule", "anthropic"):
    warnings.warn(f"Unknown LLM_MODE='{_llm_raw}', defaulting to 'rule'.", stacklevel=1)
    _llm_raw = "rule"

LLM_MODE: str = _llm_raw


# ── Execution mode ────────────────────────────────────────────────────────────
_exec_raw = os.getenv("EXECUTION_MODE", "paper").lower().strip()
if _exec_raw not in ("paper", "live", "backtest"):
    warnings.warn(f"Unknown EXECUTION_MODE='{_exec_raw}', defaulting to 'paper'.", stacklevel=1)
    _exec_raw = "paper"

EXECUTION_MODE: str = _exec_raw


# ── Paper trading parameters ─────────────────────────────────────────────────
PAPER_STARTING_CASH: float = float(os.getenv("PAPER_STARTING_CASH", "1000.0"))
PAPER_MAX_CONTRACTS: int   = int(os.getenv("PAPER_MAX_CONTRACTS", "20"))
