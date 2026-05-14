from __future__ import annotations

from typing import Sequence

from .protocol import NumericProtocol
from .state import State


class WeightedSum(NumericProtocol):
    """`process(State({"weights":[...], "values":[...]}))` → Σ wᵢxᵢ (lists must match in length)."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        weights: Sequence[float] = state.get("weights")
        values: Sequence[float] = state.get("values")

        if len(weights) != len(values):
            raise ValueError("WeightedSum: weights and values length mismatch")

        self._out = sum(w * v for w, v in zip(weights, values))

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("WeightedSum.readout before process")

        return self._out


class Mean(NumericProtocol):
    """`process(State({"list":[...]}))` → arithmetic mean, or 0 if empty."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        xs = state.f
        self._out = sum(xs) / len(xs) if xs else 0.0

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("Mean.readout before process")

        return self._out


class Median(NumericProtocol):
    """`process(State({"list":[...]}))` → median, or 0 if empty."""

    __slots__ = ("_out",)

    def __init__(self) -> None:
        self._out: float | None = None

    def process(self, state: State) -> None:
        xs = sorted(state.f)
        n = len(xs)

        if n == 0:
            self._out = 0.0
            return

        mid = n // 2
        self._out = xs[mid] if n % 2 else 0.5 * (xs[mid - 1] + xs[mid])

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("Median.readout before process")

        return self._out
