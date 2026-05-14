from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .asset_pair import AssetPair
from ._time import parse_rfc3339


@dataclass(frozen=True, slots=True)
class Trade:
    """One executed trade on Kraken's WS `trade` channel."""

    symbol: str
    side: str
    qty: float
    price: float
    ord_type: str
    trade_id: int
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Trade:
        return cls(
            symbol=str(d["symbol"]),
            side=str(d["side"]),
            qty=float(d["qty"]),
            price=float(d["price"]),
            ord_type=str(d["ord_type"]),
            trade_id=int(d["trade_id"]),
            timestamp=parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class TradeMessage:
    """Envelope around a batch of trades; snapshot returns up to ~50 most recent."""

    is_snapshot: bool
    trades: tuple[Trade, ...]

    @classmethod
    def from_kraken(cls, msg: dict[str, Any]) -> TradeMessage:
        if msg.get("channel") != "trade":
            raise ValueError(f"TradeMessage: expected channel='trade', got {msg.get('channel')!r}")

        mtype = msg.get("type")

        if mtype not in ("snapshot", "update"):
            raise ValueError(f"TradeMessage: unexpected type {mtype!r}")

        data = msg.get("data") or []
        return cls(
            is_snapshot=(mtype == "snapshot"),
            trades=tuple(Trade.from_kraken(d) for d in data),
        )

    def __iter__(self) -> Iterator[Trade]:
        return iter(self.trades)


__all__ = ["Trade", "TradeMessage", "AssetPair"]
