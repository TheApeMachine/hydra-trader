from __future__ import annotations

from typing import TYPE_CHECKING

from .protocol import NumericProtocol
from .state import State

if TYPE_CHECKING:
    from .dynamic import DynamicValue


class EMA(NumericProtocol):
    """Exponential moving average over streamed updates; all tuning comes from `span` (caller-supplied).

    Each `process` recomputes from `initial` and `updates` in order (matches `DynamicValue`).
    """

    def __init__(self, *, span: "DynamicValue") -> None:
        if span <= 0:
            raise ValueError("span must be positive")

        self._span = span
        self._k = 2.0 / (span + 1.0)
        self._ema: "DynamicValue | None" = None

    def process(self, state: State) -> None:
        from .dynamic import DynamicValue

        self._ema = state.f[0]

        for x in state.f[1:]:
            self._ema = self._k * DynamicValue(x) + (1.0 - self._k) * self._ema

    def readout(self) -> "DynamicValue":
        from .dynamic import DynamicValue

        if self._ema is None:
            raise RuntimeError("EMA.readout() called before process()")

        return DynamicValue(self._ema)
