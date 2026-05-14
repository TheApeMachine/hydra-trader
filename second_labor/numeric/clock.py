from __future__ import annotations

import math
import time
from typing import Callable


class Clock:
    """Wall-clock source; injectable for tests and replay."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[], float] | None = None) -> None:
        self._fn = fn or time.time

    def now(self) -> float:
        return self._fn()

    def set(self, fn: Callable[[], float]) -> None:
        self._fn = fn

    def reset(self) -> None:
        self._fn = time.time


class VirtualClock:
    """Monotonic non-decreasing time source advanced explicitly by replay/backtest drivers."""

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def __call__(self) -> float:
        return self._t

    def advance_to(self, target: float) -> None:
        if not math.isfinite(float(target)):
            raise ValueError(f"advance_to: expected finite target, got {target!r}")

        if target > self._t:
            self._t = float(target)
