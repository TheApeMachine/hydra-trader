from __future__ import annotations

from .dynamic import DynamicValue
from .protocol import NumericProtocol
from .state import State


class HysteresisInertia(NumericProtocol):
    """Sticky direction state in {-1, 0, +1}; flips only after `confirm_ticks` consecutive crossings.

    `process(State({"list":[value, pos_threshold, neg_threshold]}))` advances the tracker by one tick.
    """

    __slots__ = ("_confirm", "_state", "_pending_state", "_pending_count", "_out")

    def __init__(self, confirm_ticks: DynamicValue) -> None:
        self._confirm = confirm_ticks
        self._state: int = 0
        self._pending_state: int = 0
        self._pending_count: int = 0
        self._out: int | None = None

    def process(self, state: State) -> None:
        value = state.f[0]
        pos_threshold = state.f[1]
        neg_threshold = state.f[2]

        if value >= pos_threshold:
            target = 1
        elif value <= neg_threshold:
            target = -1
        else:
            target = 0

        if target == self._state:
            self._pending_state = self._state
            self._pending_count = 0
            self._out = self._state
            return

        if target == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = target
            self._pending_count = 1

        confirm = max(1, int(self._confirm.readout().f[0]))

        if self._pending_count >= confirm:
            self._state = target
            self._pending_count = 0

        self._out = self._state

    def readout(self) -> int:
        if self._out is None:
            raise RuntimeError("HysteresisInertia.readout before process")

        return self._out

    def reset(self) -> None:
        self._state = 0
        self._pending_state = 0
        self._pending_count = 0
        self._out = None
