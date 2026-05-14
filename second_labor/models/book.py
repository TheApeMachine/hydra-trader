from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ._time import parse_rfc3339


@dataclass(frozen=True, slots=True)
class BookLevel:
    """One price level. `qty == 0` in an update means delete the level."""

    price: float
    qty: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> BookLevel:
        return cls(price=float(d["price"]), qty=float(d["qty"]))


@dataclass(frozen=True, slots=True)
class Book:
    """Aggregated L2 book snapshot or delta for one symbol."""

    symbol: str
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    checksum: int
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Book:
        return cls(
            symbol=str(d["symbol"]),
            bids=tuple(BookLevel.from_kraken(x) for x in d.get("bids") or ()),
            asks=tuple(BookLevel.from_kraken(x) for x in d.get("asks") or ()),
            checksum=int(d["checksum"]),
            timestamp=parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class BookMessage:
    """Envelope around a batch of book entries; carries snapshot-vs-update intent."""

    is_snapshot: bool
    books: tuple[Book, ...]

    @classmethod
    def from_kraken(cls, msg: dict[str, Any]) -> BookMessage:
        if msg.get("channel") != "book":
            raise ValueError(f"BookMessage: expected channel='book', got {msg.get('channel')!r}")

        mtype = msg.get("type")

        if mtype not in ("snapshot", "update"):
            raise ValueError(f"BookMessage: unexpected type {mtype!r}")

        data = msg.get("data") or []
        return cls(
            is_snapshot=(mtype == "snapshot"),
            books=tuple(Book.from_kraken(d) for d in data),
        )

    def __iter__(self) -> Iterator[Book]:
        return iter(self.books)
