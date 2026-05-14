from __future__ import annotations

from typing import Any, Protocol


class Publisher(Protocol):
    def publish(self, record: dict[str, Any]) -> None: ...


class TradeFeed:
    """Pass-through forwarder: republishes raw trade rows under `kind: "trade"`."""

    def __init__(self, *, publisher: Publisher) -> None:
        self._publisher = publisher

    def measure(self, data: dict[str, Any]) -> None:
        self._publisher.publish({
            "kind": "trade",
            "symbol": data["trade.symbol"],
            "side": data["trade.side"],
            "price": data["trade.price"],
            "qty": data["trade.qty"],
            "t": data["trade.timestamp"],
        })
