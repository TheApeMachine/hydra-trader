from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from websockets.asyncio.server import ServerConnection, serve


log = logging.getLogger(__name__)

logging.getLogger("websockets.server").setLevel(logging.WARNING)


class DashBus:
    """Broadcast JSON envelopes to all connected dashboard WebSocket clients."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._clients: set[ServerConnection] = set()
        self._server_ctx = None

    async def start(self) -> None:
        self._server_ctx = await serve(self._on_client, self._host, self._port)
        log.info("dash_bus: listening on ws://%s:%d", self._host, self._port)

    async def stop(self) -> None:
        if self._server_ctx is not None:
            self._server_ctx.close()
            await self._server_ctx.wait_closed()
            self._server_ctx = None

    def publish(self, envelope: dict[str, Any]) -> None:
        if not self._clients:
            return

        payload = json.dumps(envelope, default=str)

        for client in list(self._clients):
            asyncio.create_task(self._send(client, payload))

    async def _send(self, client: ServerConnection, payload: str) -> None:
        try:
            await client.send(payload)
        except Exception:
            log.exception("dash_bus: send failed; dropping client")
            self._clients.discard(client)

    async def _on_client(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        log.info("dash_bus: client connected (%d total)", len(self._clients))

        try:
            async for _ in ws:
                pass
        except Exception:
            log.exception("dash_bus: client connection raised")
        finally:
            self._clients.discard(ws)
            log.info("dash_bus: client disconnected (%d total)", len(self._clients))
