from __future__ import annotations

import math
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
        """Advance virtual time forward only; no-op if ``target <= self.t``.

        Enforces monotonic non-decreasing ``self.t``. Raises ``ValueError`` if
        ``target`` is not a finite real number (NaN or infinity).
        """
        if not math.isfinite(float(target)):
            raise ValueError(f"advance_to: expected finite target, got {target!r}")
        if target > self.t:
            self.t = target
