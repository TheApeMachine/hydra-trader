from __future__ import annotations

import math
from typing import Sequence

from .protocol import NumericProtocol
from .state import State


class PercentReturn(NumericProtocol):
    """`process(State({"list":[first, last]}))` → (last/first - 1) * 100, or 0 if first <= 0."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        first = state.f[0]
        last = state.f[1]
        self._out = (last / first - 1.0) * 100.0 if first > 0 else 0.0

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("PercentReturn.readout before process")

        return self._out


class ReturnRatio(NumericProtocol):
    """`process(State({"list":[p0, p1, p2, floor]}))` → (p2/p1 - 1) / max(|p1/p0 - 1|, floor)."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        p0 = state.f[0]
        p1 = state.f[1]
        p2 = state.f[2]
        floor = state.f[3]

        if p0 <= 0 or p1 <= 0:
            self._out = 0.0
            return

        r_prev = p1 / p0 - 1.0
        r_now = p2 / p1 - 1.0
        self._out = r_now / max(abs(r_prev), floor)

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("ReturnRatio.readout before process")

        return self._out


class LogReturnVariance(NumericProtocol):
    """`process(State({"prices":[...]}))` → sample variance of log returns (0 if < 3 prices)."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        prices: Sequence[float] = state.get("prices")

        if len(prices) < 3:
            self._out = 0.0
            return

        rets: list[float] = []

        for i in range(1, len(prices)):
            a, b = prices[i - 1], prices[i]
            if a > 0 and b > 0:
                rets.append(math.log(b / a))

        n = len(rets)

        if n < 2:
            self._out = 0.0
            return

        m = sum(rets) / n
        var = sum((x - m) ** 2 for x in rets) / (n - 1)
        self._out = max(var, 0.0)

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("LogReturnVariance.readout before process")

        return self._out
