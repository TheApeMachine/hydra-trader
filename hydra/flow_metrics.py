"""Tape/book primitives for dynamical flow–style diagnostics.

These are observables: signed flow rate, price velocity/acceleration on the tape,
a book-depth-vs-impulse "viscosity" ratio, and return variance as a turbulence proxy.
Strategy code does not depend on this module; it feeds dashboards and logging.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

TapeRow = tuple[float, str, float, float, float]  # ts, side, price, qty, notional


def _log_return_variance(prices: Sequence[float]) -> float:
    if len(prices) < 3:
        return 0.0
    rets: list[float] = []
    for i in range(1, len(prices)):
        a, b = prices[i - 1], prices[i]
        if a > 0 and b > 0:
            rets.append(math.log(b / a))
    n = len(rets)
    if n < 2:
        return 0.0
    m = sum(rets) / n
    var = sum((x - m) ** 2 for x in rets) / (n - 1)
    return max(var, 0.0)


def _discrete_price_accel(prices: Sequence[float]) -> float:
    """Same construction as HydraTrader.maybe_micro_candidate: ratio of consecutive returns."""
    if len(prices) < 8:
        if len(prices) < 4:
            return 0.0
        p0, p1, p2 = prices[0], prices[len(prices) // 2], prices[-1]
    else:
        p0, p1, p2 = prices[-8], prices[-4], prices[-1]
    if p0 <= 0 or p1 <= 0:
        return 0.0
    r_prev = p1 / p0 - 1.0
    r_now = p2 / p1 - 1.0
    return r_now / max(abs(r_prev), 0.0005)


def compute(
    dq: Sequence[TapeRow],
    *,
    tnow: float,
    buy_not: float,
    sell_not: float,
    bid_depth_usd: float,
    ask_depth_usd: float,
) -> dict[str, float]:
    """Return normalized flow diagnostics for one symbol's current tape window."""
    out: dict[str, float] = {
        "flow_signed_usd_per_s": 0.0,
        "flow_price_vel_pct_per_s": 0.0,
        "flow_price_accel": 0.0,
        "flow_turbulence_ret_var": 0.0,
        "flow_depth_usd": 0.0,
        "flow_impulse_usd": 0.0,
        "flow_viscosity": 0.0,
        "flow_churn": 0.0,
    }
    impulse = buy_not + sell_not
    depth = max(bid_depth_usd + ask_depth_usd, 0.0)
    out["flow_depth_usd"] = depth
    out["flow_impulse_usd"] = impulse
    out["flow_viscosity"] = depth / max(impulse, 1.0)
    out["flow_churn"] = impulse / max(depth, 1.0)

    if not dq:
        return out

    t0 = dq[0][0]
    span = max(tnow - t0, 1e-3)
    signed = buy_not - sell_not
    out["flow_signed_usd_per_s"] = signed / span

    prices = [row[2] for row in dq if row[2] > 0]
    first, last = prices[0], prices[-1]
    if first > 0:
        move_pct = last / first - 1.0
        out["flow_price_vel_pct_per_s"] = (move_pct * 100.0) / span

    out["flow_price_accel"] = _discrete_price_accel(prices)
    out["flow_turbulence_ret_var"] = _log_return_variance(prices)

    return out


def format_lines(flow: dict[str, Any]) -> list[str]:
    """Human-readable lines for UI (compact)."""
    if not flow:
        return ["flow: (no tape)"]
    v = flow.get("flow_viscosity") or 0.0
    ch = flow.get("flow_churn") or 0.0
    tu = flow.get("flow_turbulence_ret_var") or 0.0
    return [
        f"flow visc depth/impulse: {v:.2f}  churn: {ch:.3f}",
        f"signed $/s: {flow.get('flow_signed_usd_per_s', 0):+.0f}  "
        f"px vel %/s: {flow.get('flow_price_vel_pct_per_s', 0):+.5f}  "
        f"accel: {flow.get('flow_price_accel', 0):+.2f}",
        f"turbulence (var log ret): {tu:.2e}",
    ]

