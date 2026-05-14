from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PeakEntry:
    """Snapshot of an active peak; payload is opaque (caller chooses what to remember)."""

    anchor: float
    seeded_at: float
    payload: dict


class PeakWatch:
    """Per-key peak tracker with expiry: anchor rises during an update window, expires on age or invalidating drop."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, PeakEntry] = {}

    def seed(self, key: str, *, now: float, anchor: float, payload: dict) -> None:
        self._entries[key] = PeakEntry(anchor=anchor, seeded_at=now, payload=dict(payload))

    def get(self, key: str) -> PeakEntry | None:
        return self._entries.get(key)

    def update_anchor(self, key: str, *, now: float, high: float, update_window_sec: float) -> None:
        entry = self._entries.get(key)

        if entry is None:
            return

        if (now - entry.seeded_at) <= update_window_sec and high > entry.anchor:
            entry.anchor = high

    def drop(self, key: str, *, close: float) -> float | None:
        entry = self._entries.get(key)

        if entry is None or entry.anchor <= 0:
            return None

        return 1.0 - close / entry.anchor

    def expire_if(self, key: str, *, now: float, max_age_sec: float, drop: float, invalidate_drop: float) -> bool:
        entry = self._entries.get(key)

        if entry is None:
            return False

        if (now - entry.seeded_at) > max_age_sec or drop >= invalidate_drop:
            del self._entries[key]
            return True

        return False

    def forget(self, key: str) -> None:
        self._entries.pop(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self._entries
