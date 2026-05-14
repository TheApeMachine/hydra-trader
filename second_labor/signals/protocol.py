from typing import Protocol


class SignalProtocol(Protocol):
    def __init__(
        self, data, computation: ComputationProtocol
    ):
        pass

    def measure(self):
        pass

    def confidence(self) -> float:
        pass


class ComputationProtocol(Protocol):
    def __init__(self, data):
        pass

    def process(self):
        pass