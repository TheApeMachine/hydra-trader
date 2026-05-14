from __future__ import annotations

from typing import Any, Protocol


class Publisher(Protocol):
    def publish(self, record: dict[str, Any]) -> None: ...


class OhlcSignal:
    """Pass-through forwarder: emits per-row OHLC records as fast as Kraken sends them."""

    def __init__(self, *, publisher: Publisher) -> None:
        self._publisher = publisher

    def measure(self, data: dict[str, Any]) -> None:
        self._publisher.publish({
            "kind": "ohlc",
            "symbol": data["ohlc.symbol"],
            "open": data["ohlc.open"],
            "high": data["ohlc.high"],
            "low": data["ohlc.low"],
            "close": data["ohlc.close"],
            "volume": data["ohlc.volume"],
            "interval": data["ohlc.interval"],
            "interval_begin": data["ohlc.interval_begin"],
        })
