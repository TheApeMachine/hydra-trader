from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class Ticker:
    """Level-1 ticker snapshot for one symbol (Kraken WS v2 `ticker` channel)."""

    symbol: str
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    last: float
    volume: float
    vwap: float
    low: float
    high: float
    change: float
    change_pct: float
    timestamp: float

    @classmethod
    def from_kraken(cls, d: dict[str, Any]) -> Ticker:
        return cls(
            symbol=str(d["symbol"]),
            bid=float(d["bid"]),
            bid_qty=float(d["bid_qty"]),
            ask=float(d["ask"]),
            ask_qty=float(d["ask_qty"]),
            last=float(d["last"]),
            volume=float(d["volume"]),
            vwap=float(d["vwap"]),
            low=float(d["low"]),
            high=float(d["high"]),
            change=float(d["change"]),
            change_pct=float(d["change_pct"]),
            timestamp=_parse_rfc3339(d["timestamp"]),
        )


@dataclass(frozen=True, slots=True)
class TickerMessage:
    """Envelope around a batch of ticker rows; carries snapshot-vs-update intent."""

    is_snapshot: bool
    tickers: tuple[Ticker, ...]

    @classmethod
    def from_kraken(cls, msg: dict[str, Any]) -> TickerMessage:
        if msg.get("channel") != "ticker":
            raise ValueError(f"TickerMessage: expected channel='ticker', got {msg.get('channel')!r}")

        mtype = msg.get("type")

        if mtype not in ("snapshot", "update"):
            raise ValueError(f"TickerMessage: unexpected type {mtype!r}")

        data = msg.get("data") or []
        return cls(
            is_snapshot=(mtype == "snapshot"),
            tickers=tuple(Ticker.from_kraken(d) for d in data),
        )

    def __iter__(self) -> Iterator[Ticker]:
        return iter(self.tickers)


def _parse_rfc3339(s: str) -> float:
    """Kraken sends timestamps like '2023-09-25T09:04:31.742648Z'."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()
