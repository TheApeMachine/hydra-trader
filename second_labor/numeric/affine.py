from __future__ import annotations

from .dynamic import DynamicValue
from .protocol import NumericProtocol
from .state import State


class IntensityAffine(NumericProtocol):
    """λ = μ + α·r with μ, α from `DynamicValue.readout()`."""

    __slots__ = ("_mu", "_alpha", "_out")

    def __init__(self, mu: DynamicValue, alpha: DynamicValue) -> None:
        self._mu = mu
        self._alpha = alpha
        self._out: float | None = None

    def process(self, state: State) -> None:
        if not self.validate([state.f, len(state.f) > 0]):
            raise ValueError("invalid state")

        self._out = self._mu.readout().f[0] + self._alpha.readout().f[0] * state.f[0]

    def readout(self) -> float:
        if not self.validate([self._out]):
            raise ValueError("invalid state")

        return self._out

    def validate(self, items: any):
        for item in items:
            if not item:
                return False

        return True