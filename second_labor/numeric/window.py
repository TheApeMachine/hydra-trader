from __future__ import annotations

from collections import deque
from typing import Iterator


class TimeWindow:
    """Bounded time-stamped sample buffer; exposes the active slice for window-cutoff readouts."""

    __slots__ = ("_samples",)

    def __init__(self, maxlen: int = 512) -> None:
        self._samples: deque[tuple[float, float]] = deque(maxlen=maxlen)

    def push(self, ts: float, value: float) -> None:
        self._samples.append((float(ts), float(value)))

    def active(self, now: float, window_sec: float) -> list[tuple[float, float]]:
        cutoff = now - window_sec
        return [(t, v) for t, v in self._samples if t >= cutoff]

    def values_within(self, now: float, window_sec: float) -> list[float]:
        cutoff = now - window_sec
        return [v for t, v in self._samples if t >= cutoff]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[tuple[float, float]]:
        return iter(self._samples)

    def clear(self) -> None:
        self._samples.clear()
