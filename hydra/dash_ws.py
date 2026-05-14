"""Push dashboard_snapshot() to all connected browsers."""
from __future__ import annotations

import os
# Raise the per-header-line limit before importing websockets. Browsers with
# many localhost cookies can send single header lines >8192 bytes; the default
# rejects them with "line too long" before our handler ever runs.
os.environ.setdefault("WEBSOCKETS_MAX_LINE_LENGTH", "65536")

import asyncio
import dataclasses
import json
import math
import threading
import time
from typing import Any

import websockets


DEFAULT_PORT = 8765
DEFAULT_INTERVAL = 0.2

CLIENTS: set[Any] = set()


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if dataclasses.is_dataclass(value):
        return {k: _sanitize(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize(v) for v in value]
    if not isinstance(value, type) and hasattr(value, "__dict__") or hasattr(value, "__slots__"):
        out = {}
        for k in dir(value):
            if k.startswith("_"):
                continue
            try:
                v = getattr(value, k)
            except Exception:
                continue
            if callable(v):
                continue
            out[k] = _sanitize(v)
        if out:
            return out
    return str(value)


async def _handler(ws):
    CLIENTS.add(ws)
    print(f"[DASH-WS] +client {ws.remote_address} (n={len(CLIENTS)})")
    try:
        async for _ in ws:
            pass
    finally:
        CLIENTS.discard(ws)
        print(f"[DASH-WS] -client {ws.remote_address} (n={len(CLIENTS)})")


async def _push(engine, optuna, params_source, interval):
    while True:
        if CLIENTS:
            try:
                opt = optuna.snapshot() if optuna is not None else {}
                snap = engine.dashboard_snapshot(focus=None, mode="LIVE", optuna=opt, params_source=params_source)
                payload = json.dumps({"ts": time.time(), "snap": _sanitize(snap)}, default=str)
                websockets.broadcast(CLIENTS, payload)
            except Exception as e:
                print(f"[DASH-WS] push error: {e}")
        await asyncio.sleep(interval)


async def _serve(engine, optuna, params_source, host, port, interval):
    async with websockets.serve(_handler, host, port):
        await _push(engine, optuna, params_source, interval)


def start_dash_ws(engine, _cfg=None, params_source="defaults", optuna_controller=None,
                  host=("0.0.0.0", "::"), port=DEFAULT_PORT, interval=DEFAULT_INTERVAL):
    def runner():
        asyncio.run(_serve(engine, optuna_controller, params_source, list(host), port, interval))
    t = threading.Thread(target=runner, name="hydra-dash-ws", daemon=True)
    t.start()
    print(f"[DASH-WS] listening on ws://{host}:{port}")
    return t
