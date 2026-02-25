"""
src/config.py

Central configuration for the p2p-trade-bot.
Reads trading parameters from the environment.
Supports .env files (via python-dotenv) but works fine without one.

Environment variables:
  PAPER_STARTING_CASH  starting cash in dollars (default 1000.0)
  PAPER_MAX_CONTRACTS  hard cap per trade (default 20)
  PAPER_DATA_DIR       where paper files live (default data/paper)
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to plain env vars


# ── Paper trading parameters ─────────────────────────────────────────────────
PAPER_STARTING_CASH: float = float(os.getenv("PAPER_STARTING_CASH", "1000.0"))
PAPER_MAX_CONTRACTS: int   = int(os.getenv("PAPER_MAX_CONTRACTS", "20"))
