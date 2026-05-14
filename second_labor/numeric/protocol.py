from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from .state import State

if TYPE_CHECKING:
    from .dynamic import DynamicValue


class NumericProtocol(Protocol):
    """Incremental numeric that consumes an initial seed and every external sample in order."""

    def process(self, state: State) -> State:
        ...

    def readout(self) -> DynamicValue:
        ...
