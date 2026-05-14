from __future__ import annotations

from .clamp import Clamp
from .protocol import NumericProtocol
from .state import State


class ScaledClamp(NumericProtocol):
    """`process(State({"list":[base, span, x, lo, hi]}))` → clamp(base · (1 − span·x), lo, hi)."""

    __slots__ = ("_clamp", "_out")

    def __init__(self) -> None:
        self._clamp = Clamp()
        self._out: float | None = None

    def process(self, state: State) -> None:
        base = state.f[0]
        span = state.f[1]
        x = state.f[2]
        lo = state.f[3]
        hi = state.f[4]

        scaled = base * (1.0 - span * x)
        self._clamp.process(State({"list": [scaled, lo, hi]}))
        self._out = self._clamp.readout()

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("ScaledClamp.readout before process")

        return self._out
