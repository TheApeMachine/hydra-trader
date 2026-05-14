from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from signals.pump import Candle


@dataclass(frozen=True, slots=True)
class ClosedCandle:
    """Emitted when a new candle interval starts and the previous one is finalized."""

    symbol: str
    minute_key: Any
    candle: Candle


class CandleStore:
    """Per-symbol OHLCV history; emits a ClosedCandle the first time a new interval is seen."""

    __slots__ = ("_history", "_maxlen")

    def __init__(self, *, maxlen: int = 180) -> None:
        self._maxlen = maxlen
        self._history: dict[str, deque[tuple[Any, Candle]]] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, *, symbol: str, minute_key: Any, candle: Candle) -> ClosedCandle | None:
        hist = self._history[symbol]
        closed: ClosedCandle | None = None

        if hist and hist[-1][0] != minute_key:
            prev_key, prev_candle = hist[-1]
            closed = ClosedCandle(symbol=symbol, minute_key=prev_key, candle=prev_candle)
            hist.append((minute_key, candle))
        elif hist:
            hist[-1] = (minute_key, candle)
        else:
            hist.append((minute_key, candle))

        return closed

    def history(self, symbol: str) -> list[Candle]:
        return [c for _, c in self._history.get(symbol, ())]

    def closed_history(self, symbol: str) -> list[Candle]:
        hist = self._history.get(symbol)

        if not hist or len(hist) < 2:
            return []

        return [c for _, c in list(hist)[:-1]]

    def latest(self, symbol: str) -> Candle | None:
        hist = self._history.get(symbol)
        return hist[-1][1] if hist else None

    def last_price(self, symbol: str) -> float:
        candle = self.latest(symbol)
        return candle.close if candle else 0.0

    def clear(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._history.clear()
        else:
            self._history.pop(symbol, None)
