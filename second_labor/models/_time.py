from __future__ import annotations

from datetime import datetime, timezone


def parse_rfc3339(s: str) -> float:
    """Kraken sends timestamps like '2023-09-25T09:04:31.742648Z' or with explicit offset."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    return datetime.fromisoformat(s).astimezone(timezone.utc).timestamp()
