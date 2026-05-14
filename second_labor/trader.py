from typing import Protocol


class Trader(Protocol):
    def __init__(self):
        pass

    def trade(self):
        pass