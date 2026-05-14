from __future__ import annotations

import math

from .protocol import NumericProtocol
from .state import State


class Clamp(NumericProtocol):
    """`process(State({"list":[value, lo, hi]}))` → max(lo, min(hi, value))."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        value = state.f[0]
        lo = state.f[1]
        hi = state.f[2]
        self._out = max(lo, min(hi, value))

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("Clamp.readout before process")

        return self._out


class LogClamped(NumericProtocol):
    """`process(State({"list":[value, lo, hi, eps]}))` → clamp(log(max(value, eps)), lo, hi); 0 if value <= 0."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        value = state.f[0]
        lo = state.f[1]
        hi = state.f[2]
        eps = state.f[3]

        if value <= 0:
            self._out = lo
            return

        self._out = max(lo, min(hi, math.log(max(value, eps))))

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("LogClamped.readout before process")

        return self._out
