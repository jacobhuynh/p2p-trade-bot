"""
src/web/decision_store.py

Persists each completed pipeline decision (GAME_WINNER or PLAYER_PROP) as a
JSON file under data/decisions/.  The file is what the Trade Detail page
reads to reconstruct the per-agent workflow visualization.

Filename:  {ts_ms}_{status}_{ticker}.json
Content:   { id, ts, ticker, market_type, status, trade_packet, decision }
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

_BASE = Path("data/decisions")
_BASE.mkdir(parents=True, exist_ok=True)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_ticker(t: str) -> str:
    return _SAFE.sub("-", t)[:80] or "UNKNOWN"


def save_decision(
    *,
    market_type: str,
    status: str,
    trade_packet: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Persist a decision and return the record (with assigned id)."""
    ticker = trade_packet.get("ticker") or decision.get("ticker") or "UNKNOWN"
    ts_ms = int(time.time() * 1000)
    decision_id = f"{ts_ms}-{uuid.uuid4().hex[:8]}"
    record = {
        "id": decision_id,
        "ts": ts_ms / 1000,
        "ticker": ticker,
        "market_type": market_type,
        "status": status,
        "trade_packet": trade_packet,
        "decision": decision,
    }
    fname = f"{ts_ms}_{status}_{_safe_ticker(ticker)}.json"
    path = _BASE / fname
    path.write_text(json.dumps(record, default=str, indent=2), encoding="utf-8")
    record["_file"] = path.name
    return record


def list_decisions(limit: int = 200) -> list[dict[str, Any]]:
    """Return decision summaries (newest first), without the heavy fields."""
    files = sorted(_BASE.glob("*.json"), reverse=True)[:limit]
    out: list[dict[str, Any]] = []
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        d = r.get("decision", {}) or {}
        q = d.get("quant_summary", {}) or {}
        c = d.get("critic", {}) or {}
        out.append({
            "id": r.get("id"),
            "file": f.name,
            "ts": r.get("ts"),
            "ticker": r.get("ticker"),
            "market_type": r.get("market_type"),
            "status": r.get("status"),
            "action": d.get("action"),
            "price": d.get("price"),
            "confidence": d.get("confidence"),
            "edge": d.get("edge"),
            "calibration_gap": q.get("calibration_gap"),
            "risk_score": c.get("risk_score"),
            "veto_reason": c.get("veto_reason"),
        })
    return out


def get_decision(decision_id: str) -> dict[str, Any] | None:
    """Look up the full decision record by id."""
    for f in _BASE.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("id") == decision_id:
            r["_file"] = f.name
            return r
    return None


def pick_recent(market_type: str) -> dict[str, Any] | None:
    """Return the newest decision record matching market_type, or None."""
    for f in sorted(_BASE.glob("*.json"), reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("market_type") == market_type:
            r["_file"] = f.name
            return r
    return None
