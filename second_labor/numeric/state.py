from typing import Optional


class State:
    def __init__(self, initial: Optional[dict]):
        self.d: dict = initial or {}

    @property
    def f(self) -> list:
        """`list` payloads (one float → one element). Key matches the type name."""
        return self.d.get("list", [])

    def update(self, d: dict) -> dict:
        self.d = {
            **self.d,
            **d
        }

        return self.d

    def get(self, key):
        if key not in self.d:
            raise ValueError(f"key: {key} not found")

        return self.d[key]