from __future__ import annotations

from enum import Enum

from .ema import EMA
from .state import State


class NOOP:
    @staticmethod
    def process(state: State) -> State:
        return state

    @staticmethod
    def readout() -> None:
        return None


class DynamicValueType(Enum):
    """Picks which `NumericProtocol` runs inside a `DynamicValue`.

    Members are *only* implementation tags — no coefficients or “magic numbers”.
    E.g. `EMA` still needs `span=…` (and `initial=…`) when you construct `DynamicValue`."""
    NOOP = NOOP
    EMA = EMA


class DynamicValue:
    def __init__(
        self, *, t: DynamicValueType = NOOP, initial: State,
    ) -> None:
        self._t = t
        self._state = initial

    def update(self, state: State):
        self._state.update(state)

    def readout(self) -> State:
        self._t.process(self._state)
        return self._state
