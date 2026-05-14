from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ._time import parse_rfc3339


@dataclass(frozen=True, slots=True)
class Level3Order:
    """Order-by-order entry as it appears in a level3 snapshot."""

    order_id: str
    limit_price: float
    order_qty: float
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Level3Order:
        return cls(
            order_id=str(d["order_id"]),
            limit_price=float(d["limit_price"]),
            order_qty=float(d["order_qty"]),
            timestamp=parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class Level3Event:
    """Order-by-order update entry; `event` ∈ {add, modify, delete}."""

    event: str
    order_id: str
    limit_price: float
    order_qty: float
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Level3Event:
        return cls(
            event=str(d["event"]),
            order_id=str(d["order_id"]),
            limit_price=float(d["limit_price"]),
            order_qty=float(d["order_qty"]),
            timestamp=parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class Level3Snapshot:
    """Full L3 snapshot for one symbol."""

    symbol: str
    bids: tuple[Level3Order, ...]
    asks: tuple[Level3Order, ...]
    checksum: int
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Level3Snapshot:
        return cls(
            symbol=str(d["symbol"]),
            bids=tuple(Level3Order.from_kraken(x) for x in d.get("bids") or ()),
            asks=tuple(Level3Order.from_kraken(x) for x in d.get("asks") or ()),
            checksum=int(d["checksum"]),
            timestamp=parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class Level3Update:
    """Delta update for one symbol; `bids`/`asks` are event lists, not order lists."""

    symbol: str
    bids: tuple[Level3Event, ...]
    asks: tuple[Level3Event, ...]
    checksum: int
    timestamp: float | None

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Level3Update:
        ts = d.get("timestamp")
        return cls(
            symbol=str(d["symbol"]),
            bids=tuple(Level3Event.from_kraken(x) for x in d.get("bids") or ()),
            asks=tuple(Level3Event.from_kraken(x) for x in d.get("asks") or ()),
            checksum=int(d["checksum"]),
            timestamp=parse_rfc3339(ts) if isinstance(ts, str) else None,
        )


@dataclass(frozen=True, slots=True)
class Level3Message:
    """Envelope around a batch of L3 entries. `is_snapshot` selects which payload tuple is populated."""

    is_snapshot: bool
    snapshots: tuple[Level3Snapshot, ...]
    updates: tuple[Level3Update, ...]

    @classmethod
    def from_kraken(cls, msg: dict[str, Any]) -> Level3Message:
        if msg.get("channel") != "level3":
            raise ValueError(f"Level3Message: expected channel='level3', got {msg.get('channel')!r}")

        mtype = msg.get("type")
        data = msg.get("data") or []

        if mtype == "snapshot":
            return cls(
                is_snapshot=True,
                snapshots=tuple(Level3Snapshot.from_kraken(d) for d in data),
                updates=(),
            )

        if mtype == "update":
            return cls(
                is_snapshot=False,
                snapshots=(),
                updates=tuple(Level3Update.from_kraken(d) for d in data),
            )

        raise ValueError(f"Level3Message: unexpected type {mtype!r}")

    def __iter__(self) -> Iterator[Level3Snapshot | Level3Update]:
        return iter(self.snapshots if self.is_snapshot else self.updates)
