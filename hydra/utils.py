from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone


def sf(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def median(xs, default: float = 0.0) -> float:
    vals = [x for x in xs if x is not None]
    
    if not vals:
        return default
    
    return statistics.median(vals)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def close_position(candle: tuple) -> float:
    rng = candle[2] - candle[3]

    if rng <= 0:
        return 0.5
    
    return (candle[4] - candle[3]) / rng


def candle_close_pos_dict(c: dict) -> float:
    rng = c["high"] - c["low"]
    
    if rng <= 0:
        return 0.5
    
    return (c["close"] - c["low"]) / rng


def breakeven_move_pct(cfg) -> float:
    return (1 + cfg.slippage) / ((1 - cfg.slippage) * ((1 - cfg.fee_pct) ** 2)) - 1.0


class InertiaTracker:
    """Sticky direction state with hysteresis.

    Holds a state in {-1, 0, +1}. Flipping requires the raw value to cross
    the relevant threshold AND stay across it for ``confirm_ticks`` consecutive
    updates. Single-tick spikes do nothing — the state has momentum.

    Used for display/gate signals only; never for instantaneous triggers like
    stop-loss or pump detection.
    """

    __slots__ = ("state", "_pending_state", "_pending_count", "_confirm")

    def __init__(self, confirm_ticks: int = 4):
        self.state: int = 0
        self._pending_state: int = 0
        self._pending_count: int = 0
        self._confirm = max(1, int(confirm_ticks))

    def update(self, value: float, pos_threshold: float, neg_threshold: float) -> int:
        if value >= pos_threshold:
            target = 1
        elif value <= neg_threshold:
            target = -1
        else:
            target = 0
        if target == self.state:
            self._pending_state = self.state
            self._pending_count = 0
            return self.state
        if target == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = target
            self._pending_count = 1
        if self._pending_count >= self._confirm:
            self.state = target
            self._pending_count = 0
        return self.state

    def reset(self) -> None:
        self.state = 0
        self._pending_state = 0
        self._pending_count = 0


def iso_from_ts(t: float) -> str:
    if not isinstance(t, (int, float)) or not math.isfinite(float(t)):
        raise ValueError(f"iso_from_ts: expected finite numeric timestamp, got {t!r}")
    try:
        return datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
    except (ValueError, OSError) as e:
        raise ValueError(f"iso_from_ts: fromtimestamp failed for {t!r}") from e
