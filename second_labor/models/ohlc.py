from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from ._time import parse_rfc3339


@dataclass(frozen=True, slots=True)
class Ohlc:
    """One OHLCV bucket; bucket identified by (symbol, interval, interval_begin)."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    vwap: float
    trades: float
    volume: float
    interval_begin: float
    interval: int
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Ohlc:
        return cls(
            symbol=str(d["symbol"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            vwap=float(d["vwap"]),
            trades=float(d["trades"]),
            volume=float(d["volume"]),
            interval_begin=parse_rfc3339(d["interval_begin"]),
            interval=int(d["interval"]),
            timestamp=parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class OhlcMessage:
    """Envelope around a batch of OHLC rows; carries snapshot-vs-update intent."""

    is_snapshot: bool
    timestamp: float | None
    bars: tuple[Ohlc, ...]

    @classmethod
    def from_kraken(cls, msg: dict[str, Any]) -> OhlcMessage:
        if msg.get("channel") != "ohlc":
            raise ValueError(f"OhlcMessage: expected channel='ohlc', got {msg.get('channel')!r}")

        mtype = msg.get("type")

        if mtype not in ("snapshot", "update"):
            raise ValueError(f"OhlcMessage: unexpected type {mtype!r}")

        ts = msg.get("timestamp")
        data = msg.get("data") or []
        return cls(
            is_snapshot=(mtype == "snapshot"),
            timestamp=parse_rfc3339(ts) if isinstance(ts, str) else None,
            bars=tuple(Ohlc.from_kraken(d) for d in data),
        )

    def __iter__(self) -> Iterator[Ohlc]:
        return iter(self.bars)
