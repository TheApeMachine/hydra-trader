from __future__ import annotations

import math

from .dynamic import DynamicValue
from .protocol import NumericProtocol
from .state import State


class PostEventJump(NumericProtocol):
    """Standard exponential kernel: each event adds unit weight → r' = r + 1 before the next decay."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: DynamicValue | None = None

    def process(self, state: State) -> None:
        self._out = state.f[0] + 1.0

    def readout(self) -> float:
        return self._out


class ExpectedEvents(NumericProtocol):
    """∫₀ᴴ λ(u) du with λ(u)=μ + α·r·exp(−βu); pass H as `process(r_now, [H])`."""

    __slots__ = ("_mu", "_alpha", "_beta", "_out")

    def __init__(self, mu: DynamicValue, alpha: DynamicValue, beta: DynamicValue) -> None:
        self._mu = mu
        self._alpha = alpha
        self._beta = beta
        self._out: DynamicValue | None = None

    def process(self, state: State) -> None:
        r_now = state.f[0]
        h = state.f[1]
        
        if h <= 0:
            raise ValueError("horizon_sec must be positive")
        
        mu = self._mu.readout().f[0]
        a = self._alpha.readout().f[0]
        b = self._beta.readout().f[0]
        burst = a * r_now * (1.0 - math.exp(-b * h)) / b
        self._out = mu * h + burst

    def readout(self) -> DynamicValue:
        if self._out is None:
            raise RuntimeError("ExpKernelForecastExpectedEvents.readout before process")
        
        return self._out
