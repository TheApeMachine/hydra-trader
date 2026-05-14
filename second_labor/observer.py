from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

import requests
import websockets

from .models.asset_pair import AssetPair
from .models.book import BookMessage
from .models.instrument import InstrumentMessage
from .models.level3 import Level3Message
from .models.ohlc import OhlcMessage
from .models.ticker import TickerMessage
from .models.trade import TradeMessage


KRAKEN_BASE_URL = "https://api.kraken.com/0/public"
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

log = logging.getLogger(__name__)


SignalCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


_PARSERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "ticker": TickerMessage.from_kraken,
    "book": BookMessage.from_kraken,
    "level3": Level3Message.from_kraken,
    "ohlc": OhlcMessage.from_kraken,
    "trade": TradeMessage.from_kraken,
    "instrument": InstrumentMessage.from_kraken,
}

_ROW_ATTRS: dict[str, str] = {
    "ticker": "tickers",
    "book": "books",
    "ohlc": "bars",
    "trade": "trades",
}


@dataclass(slots=True)
class Subscription:
    """One signal's interest: `channel.field` keys + symbol list + per-channel extras + the callback to invoke per row."""

    keys: tuple[str, ...]
    symbols: tuple[str, ...]
    callback: SignalCallback
    channel_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    channels: frozenset[str] = field(init=False)
    paths_by_channel: dict[str, tuple[tuple[str, str], ...]] = field(init=False)

    def __post_init__(self) -> None:
        by_channel: dict[str, list[tuple[str, str]]] = {}

        for k in self.keys:
            if "." not in k:
                raise ValueError(f"Subscription key must be '<channel>.<field>'; got {k!r}")

            channel, path = k.split(".", 1)
            by_channel.setdefault(channel, []).append((k, path))

        self.channels = frozenset(by_channel)
        self.paths_by_channel = {ch: tuple(items) for ch, items in by_channel.items()}

        for ch in self.channel_params:
            if ch not in self.channels:
                raise ValueError(f"Subscription extras given for channel {ch!r} not in keys")

    def subscribe_frames(self, req_id_start: int) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []

        for offset, channel in enumerate(sorted(self.channels)):
            params: dict[str, Any] = {"channel": channel, "symbol": list(self.symbols)}
            params.update(self.channel_params.get(channel, {}))

            frames.append({
                "method": "subscribe",
                "params": params,
                "req_id": req_id_start + offset,
            })

        return frames


class Observer:
    """REST bootstrap + WS pump. Parses messages, extracts requested fields, dispatches to subscribed signals."""

    def __init__(self) -> None:
        self.conn = websockets.connect
        self.asset_pairs: list[AssetPair] = []
        self.focused_pairs: list[AssetPair] = []
        self.session = requests.Session()
        self.timeout = 30
        self.subscriptions: list[Subscription] = []

    def subscribe(
        self,
        keys: Sequence[str],
        callback: SignalCallback,
        symbols: Sequence[str],
        channel_params: dict[str, dict[str, Any]] | None = None,
    ) -> Subscription:
        sub = Subscription(
            keys=tuple(keys),
            symbols=tuple(symbols),
            callback=callback,
            channel_params=dict(channel_params or {}),
        )
        self.subscriptions.append(sub)
        return sub

    async def run(self) -> None:
        self.asset_pairs = [
            AssetPair.from_kraken(pair_name, info)
            for pair_name, info in self.kraken().items()
        ]

        async with self.conn(KRAKEN_WS_URL) as ws:
            req_id = 0

            for sub in self.subscriptions:
                for frame in sub.subscribe_frames(req_id):
                    await ws.send(json.dumps(frame))
                    req_id += 1

            async for raw in ws:
                await self._handle(raw)

    async def _handle(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as e:
            log.error("observer: JSON decode failed: %s; raw=%r", e, raw[:200])
            return

        if not isinstance(msg, dict):
            log.warning("observer: top-level message is not an object: %r", msg)
            return

        if "method" in msg:
            self._log_subscribe_response(msg)
            return

        channel = msg.get("channel")

        if channel in (None, "heartbeat", "status"):
            return

        parser = _PARSERS.get(channel)

        if parser is None:
            log.info("observer: no parser for channel %r; dropping", channel)
            return

        try:
            parsed = parser(msg)
        except (KeyError, ValueError, TypeError) as e:
            log.exception("observer: parser for channel=%r failed: %s", channel, e)
            return

        for row in self._iter_rows(channel, parsed):
            for sub in self.subscriptions:
                if channel not in sub.channels:
                    continue

                payload = self._extract(row, sub.paths_by_channel[channel])

                try:
                    result = sub.callback(payload)

                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    log.exception(
                        "observer: callback raised on channel=%r keys=%r; continuing",
                        channel,
                        sub.keys,
                    )

    @staticmethod
    def _log_subscribe_response(msg: dict[str, Any]) -> None:
        method = msg.get("method")
        success = msg.get("success")
        req_id = msg.get("req_id")

        if success is False:
            log.error("observer: %s req_id=%s failed: %s", method, req_id, msg.get("error") or msg)
        else:
            log.info("observer: %s req_id=%s ok (%s)", method, req_id, msg.get("result") or "")

    def kraken(self) -> dict[str, dict[str, Any]]:
        response = self.session.get(
            f"{KRAKEN_BASE_URL}/AssetPairs",
            params={},
            timeout=self.timeout,
        )
        body = response.json()

        if err := body.get("error") or []:
            raise RuntimeError(f"Kraken API error: {err}")

        out = body.get("result")

        if not isinstance(out, dict):
            raise RuntimeError("Kraken AssetPairs: missing result")

        return out

    def _iter_rows(self, channel: str, parsed: Any) -> Iterable[Any]:
        attr = _ROW_ATTRS.get(channel)

        if attr is None:
            yield parsed
            return

        yield from getattr(parsed, attr)

    def _extract(self, row: Any, paths: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for key, path in paths:
            out[key] = self._resolve(row, path)

        return out

    def _resolve(self, obj: Any, path: str) -> Any:
        current = obj

        for part in path.split("."):
            current = getattr(current, part)

        return current
