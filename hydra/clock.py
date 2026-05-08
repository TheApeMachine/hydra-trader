from __future__ import annotations

import time
from typing import Callable


class Clock:
    def __init__(self, fn: Callable[[], float] | None = None):
        self._fn = fn or time.time

    def now(self) -> float:
        return self._fn()

    def set(self, fn: Callable[[], float]) -> None:
        self._fn = fn

    def reset(self) -> None:
        self._fn = time.time


class VirtualClock:
    """Monotonic virtual clock advanced explicitly by replay/backtest engines."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance_to(self, target: float) -> None:
        if target > self.t:
            self.t = target



