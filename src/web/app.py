"""
src/web/app.py

FastAPI backend for the live visualization frontend.

Run:
    uvicorn src.web.app:app --reload --port 8000

Endpoints
---------
GET  /api/health                  liveness + pipeline status
GET  /api/decisions               recent pipeline decisions (summaries)
GET  /api/decisions/{id}          full decision JSON (workflow viz reads this)
GET  /api/trades                  rows from data/live_trades.db (?status=open|evaluated|all)
GET  /api/trades/summary          aggregate P&L from TradeLogger.summary()
POST /api/settle                  run src/settle.py and stream stdout (text/plain)
WS   /ws                          live pipeline events
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sys
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.execution.trade_logger import TradeLogger
from src.web import decision_store
from src.web.events import bus

_PIPELINE_STATE: dict = {"running": False, "error": None}


async def _run_pipeline_forever() -> None:
    """Run the Kalshi WebSocket client in this process so events flow into
    the bus.  If credentials are missing we record the error and exit —
    the frontend still works with historical data."""
    try:
        from src.pipeline.websocket_client import KalshiWebsocketClient
    except Exception as e:
        _PIPELINE_STATE["error"] = f"import failed: {e}"
        return

    if not os.getenv("KALSHI_API_KEY_ID") or not os.getenv("KALSHI_PRIVATE_KEY_PATH"):
        _PIPELINE_STATE["error"] = "KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH not set"
        return

    client = KalshiWebsocketClient()
    _PIPELINE_STATE["running"] = True
    try:
        await client.run_forever()
    except asyncio.CancelledError:
        raise
    except Exception as e:
        _PIPELINE_STATE["error"] = str(e)
    finally:
        _PIPELINE_STATE["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_run_pipeline_forever())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="p2p-trade-bot viz", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "pipeline_running": _PIPELINE_STATE["running"],
        "pipeline_error": _PIPELINE_STATE["error"],
    }


@app.get("/api/decisions")
async def list_decisions(limit: int = 200) -> list[dict]:
    return decision_store.list_decisions(limit=limit)


@app.get("/api/decisions/{decision_id}")
async def get_decision(decision_id: str) -> dict:
    rec = decision_store.get_decision(decision_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="decision not found")
    return rec


@app.get("/api/trades")
async def list_trades(status: str = "all") -> list[dict]:
    logger = TradeLogger()
    if status == "open":
        return logger.open_trades()
    if status == "evaluated":
        return logger.evaluated_trades()
    return logger.open_trades() + logger.evaluated_trades()


@app.get("/api/trades/summary")
async def trades_summary() -> dict:
    return TradeLogger().summary()


@app.post("/api/settle")
async def run_settle() -> StreamingResponse:
    """Stream stdout from src.settle.run_settle() back to the browser."""
    from src.settle import run_settle as _run

    async def stream():
        loop = asyncio.get_event_loop()
        buffer = io.StringIO()

        def _do() -> str:
            with redirect_stdout(buffer):
                _run()
            return buffer.getvalue()

        text = await loop.run_in_executor(None, _do)
        for line in text.splitlines(keepends=True):
            yield line

    return StreamingResponse(stream(), media_type="text/plain")


# ── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    q = bus.subscribe()
    try:
        # Replay recent history so a freshly opened tab has context.
        for event in bus.history():
            await websocket.send_text(json.dumps(event, default=str))
        while True:
            event = await q.get()
            await websocket.send_text(json.dumps(event, default=str))
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(q)
