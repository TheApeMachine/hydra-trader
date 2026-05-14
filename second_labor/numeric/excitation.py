from __future__ import annotations

import math

from .dynamic import DynamicValue
from .protocol import NumericProtocol
from .state import State


class DecayedExcitation(NumericProtocol):
    """Decay branching sum toward an evaluation time: r(now) from (r, t_last, β)."""

    __slots__ = ("_beta", "_t_last", "_out")

    def __init__(self, beta: DynamicValue) -> None:
        self._beta = beta
        self._t_last: float | None = None
        self._out: float | None = None

    def set_t_last(self, t_last: float | None) -> None:
        self._t_last = t_last

    def process(self, state: State) -> None:
        r = state.f[0]
        now = state.f[1]
        b = self._beta.readout().f[0]

        if self._t_last is None:
            self._out = r
        else:
            dt = now - self._t_last

            if dt < 0:
                raise ValueError("evaluation time cannot go backwards")

            self._out = r * math.exp(-b * dt)

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("DecayedExcitation.readout before process")

        return self._out
