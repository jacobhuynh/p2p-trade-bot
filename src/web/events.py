"""
src/web/events.py

In-process pub/sub for streaming pipeline events to WebSocket clients.

The pipeline (websocket_client.handle_message) calls bus.emit(...) at each
boundary; the FastAPI WebSocket endpoint subscribes and fans every event
out to connected browsers as JSON.

If nobody is subscribed, emit() is a cheap no-op so the standalone
`python -m src.pipeline.websocket_client` invocation pays no cost.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any


class EventBus:
    def __init__(self, history_size: int = 200) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._history: list[dict[str, Any]] = []
        self._history_size = history_size

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event. Safe to call from any thread / sync context."""
        event = {
            "id": uuid.uuid4().hex,
            "type": event_type,
            "ts": time.time(),
            "data": data,
        }
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]

        if not self._subscribers:
            return

        loop = self._loop_or_none()
        if loop is None:
            return

        for q in list(self._subscribers):
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except (RuntimeError, asyncio.QueueFull):
                pass

    @staticmethod
    def _loop_or_none() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            return None


bus = EventBus()
