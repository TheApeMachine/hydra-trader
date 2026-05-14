from __future__ import annotations

from .protocol import NumericProtocol
from .state import State


class Ratio(NumericProtocol):
    """`process(State({"list":[num, den, den_floor]}))` → num / max(den, den_floor)."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        num = state.f[0]
        den = state.f[1]
        floor = state.f[2]
        self._out = num / max(den, floor)

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("Ratio.readout before process")

        return self._out


class PerSecond(NumericProtocol):
    """`process(State({"list":[value, t_now, t_first, span_floor]}))` → value / max(t_now - t_first, floor)."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        value = state.f[0]
        t_now = state.f[1]
        t_first = state.f[2]
        floor = state.f[3]
        self._out = value / max(t_now - t_first, floor)

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("PerSecond.readout before process")

        return self._out
