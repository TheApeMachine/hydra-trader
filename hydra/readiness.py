"""Single source of truth for the per-symbol Buy/Sell readiness score.

Used by the matplotlib dashboard, the websocket snapshot (so the web dashboard
sees the same number), and any internal heuristic that wants the gauge value.
"""

from __future__ import annotations

import math


def symbol_readiness_score(sym: str, snap) -> float:
    """Compute a per-symbol readiness score in [-100, +100]."""
    cand_score = 0.0

    for c in snap.get("candidates") or ():
        if c.get("symbol") == sym:
            s = float(c.get("score") or 0.0)

            if s > cand_score:
                cand_score = s

    cand_contrib = min(cand_score / 10.0 * 50.0, 50.0)

    view = snap.get("market")
    excitation = 0.0
    ratio = 1.0
    mh = getattr(view, "multi_hawkes_history", None) or {}
    series = mh.get(sym)

    if series:
        _, excitation, ratio, _slope = series[-1]

    if ratio <= 0:
        direction = -2.0
    else:
        direction = math.log(max(ratio, 1e-9))
        direction = max(-2.0, min(2.0, direction))

    excess_excitation = max(0.0, float(excitation) - 1.0)
    excitation_sign = 1.0 if direction > 0 else -1.0 if direction < 0 else 0.0
    hawkes_mag_contrib = excess_excitation * 12.0 * excitation_sign
    hawkes_dir_contrib = direction * 18.0

    btc_nudge = 0.0

    if snap.get("btc_flush"):
        btc_nudge = -25.0
    elif not snap.get("btc_ok"):
        btc_nudge = -8.0

    score = cand_contrib + hawkes_mag_contrib + hawkes_dir_contrib + btc_nudge
    return max(-100.0, min(100.0, score))
