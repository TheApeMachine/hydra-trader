from __future__ import annotations

from .dynamic import DynamicValue
from .protocol import NumericProtocol
from .state import State


class Cooldown(NumericProtocol):
    """Time-based gate: `process(State({"list":[now]}))` → 1.0 if elapsed >= window, else 0.0.

    Call `arm(now)` after acting on a positive readout to start a new cooldown window.
    `window_sec` is a `DynamicValue` so the cooldown can be tuned externally.
    """

    __slots__ = ("_window", "_armed_at", "_out")

    def __init__(self, window_sec: DynamicValue) -> None:
        self._window = window_sec
        self._armed_at: float | None = None
        self._out: float | None = None

    def process(self, state: State) -> None:
        now = state.f[0]

        if self._armed_at is None:
            self._out = 1.0
            return

        window = self._window.readout().f[0]
        self._out = 1.0 if (now - self._armed_at) >= window else 0.0

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("Cooldown.readout before process")

        return self._out

    def arm(self, now: float) -> None:
        self._armed_at = float(now)

    def reset(self) -> None:
        self._armed_at = None
        self._out = None
