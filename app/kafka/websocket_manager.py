"""
WebSocket connection manager.

Tracks active WebSocket connections per job_id.
Consumer workers call broadcast() when job status changes,
pushing real-time updates to connected clients.

Usage (client):
    ws = new WebSocket("ws://localhost:8000/api/v1/ws/{job_id}")
    ws.onmessage = (e) => console.log(JSON.parse(e.data))
"""
import json
from collections import defaultdict
from fastapi import WebSocket
from loguru import logger


class WebSocketManager:
    def __init__(self):
        # job_id → set of active WebSocket connections
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[job_id].add(ws)
        logger.info(f"WS connected: job={job_id} total={len(self._connections[job_id])}")

    def disconnect(self, job_id: str, ws: WebSocket) -> None:
        self._connections[job_id].discard(ws)
        if not self._connections[job_id]:
            del self._connections[job_id]
        logger.info(f"WS disconnected: job={job_id}")

    async def broadcast(self, job_id: str, payload: dict) -> None:
        """Send payload to all clients watching this job_id."""
        connections = self._connections.get(job_id, set()).copy()
        if not connections:
            return
        message = json.dumps(payload)
        dead = set()
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(job_id, ws)


# Global singleton — imported by both routes and consumer
ws_manager = WebSocketManager()