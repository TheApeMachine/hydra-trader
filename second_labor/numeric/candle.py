from __future__ import annotations

from .protocol import NumericProtocol
from .state import State


class RangePercent(NumericProtocol):
    """`process(State({"list":[high, low]}))` → (high - low) / low * 100, or 0 if low <= 0."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        high = state.f[0]
        low = state.f[1]
        self._out = (high - low) / low * 100.0 if low > 0 else 0.0

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("RangePercent.readout before process")

        return self._out


class ClosePosition(NumericProtocol):
    """`process(State({"list":[high, low, close]}))` → (close - low) / (high - low), or 0.5 on degenerate range."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        high = state.f[0]
        low = state.f[1]
        close = state.f[2]
        rng = high - low
        self._out = (close - low) / rng if rng > 0 else 0.5

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("ClosePosition.readout before process")

        return self._out
