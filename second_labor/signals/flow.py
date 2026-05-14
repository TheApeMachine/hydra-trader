from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from numeric.dynamic import DynamicValue
from numeric.ratio import Ratio, PerSecond
from numeric.returns import PercentReturn, ReturnRatio, LogReturnVariance
from numeric.state import State

TapeRow = tuple[float, str, float, float, float]


@dataclass(frozen=True, slots=True)
class FlowReadout:
    signed_usd_per_s: float
    price_vel_pct_per_s: float
    price_accel: float
    turbulence_ret_var: float
    depth_usd: float
    impulse_usd: float
    viscosity: float
    churn: float


class FlowSignal:
    """Tape/book flow diagnostics composed from atomic numerics."""

    def __init__(self) -> None:
        self._viscosity = Ratio()
        self._churn = Ratio()
        self._signed_rate = PerSecond()
        self._price_vel = PercentReturn()
        self._price_vel_rate = PerSecond()
        self._price_accel = ReturnRatio()
        self._turbulence = LogReturnVariance()
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_readout: FlowReadout | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def measure(
        self,
        dq: Sequence[TapeRow],
        *,
        tnow: float,
        buy_not: float,
        sell_not: float,
        bid_depth_usd: float,
        ask_depth_usd: float,
    ) -> FlowReadout:
        impulse = buy_not + sell_not
        depth = max(bid_depth_usd + ask_depth_usd, 0.0)

        self._viscosity.process(State({"list": [depth, impulse, 1.0]}))
        self._churn.process(State({"list": [impulse, depth, 1.0]}))

        if not dq:
            out = FlowReadout(
                signed_usd_per_s=0.0,
                price_vel_pct_per_s=0.0,
                price_accel=0.0,
                turbulence_ret_var=0.0,
                depth_usd=depth,
                impulse_usd=impulse,
                viscosity=self._viscosity.readout(),
                churn=self._churn.readout(),
            )
            self._last_readout = out
            return out

        t_first = dq[0][0]
        signed = buy_not - sell_not

        self._signed_rate.process(State({"list": [signed, tnow, t_first, 1e-3]}))

        prices = [row[2] for row in dq if row[2] > 0]
        first, last = prices[0], prices[-1]

        self._price_vel.process(State({"list": [first, last]}))
        move_pct = self._price_vel.readout()
        self._price_vel_rate.process(State({"list": [move_pct, tnow, t_first, 1e-3]}))

        p0, p1, p2 = self._accel_anchors(prices)
        self._price_accel.process(State({"list": [p0, p1, p2, 5e-4]}))
        self._turbulence.process(State({"prices": prices}))

        out = FlowReadout(
            signed_usd_per_s=self._signed_rate.readout(),
            price_vel_pct_per_s=self._price_vel_rate.readout(),
            price_accel=self._price_accel.readout(),
            turbulence_ret_var=self._turbulence.readout(),
            depth_usd=depth,
            impulse_usd=impulse,
            viscosity=self._viscosity.readout(),
            churn=self._churn.readout(),
        )
        self._last_readout = out
        return out

    @staticmethod
    def _accel_anchors(prices: Sequence[float]) -> tuple[float, float, float]:
        if len(prices) < 4:
            return 0.0, 0.0, 0.0

        if len(prices) < 8:
            return prices[0], prices[len(prices) // 2], prices[-1]

        return prices[-8], prices[-4], prices[-1]
