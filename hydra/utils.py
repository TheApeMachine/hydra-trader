from __future__ import annotations

import statistics
from datetime import datetime


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


def iso_from_ts(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat()



