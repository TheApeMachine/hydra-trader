from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from numeric.blend import WeightedSum
from numeric.clamp import Clamp
from numeric.cooldown import Cooldown
from numeric.dynamic import DynamicValue
from numeric.ratio import Ratio
from numeric.returns import ReturnRatio
from numeric.state import State


@dataclass(frozen=True, slots=True)
class MicroSnapshot:
    """Per-symbol microstructure features feeding the ignition signal."""

    trades: int
    buy_not: float
    sell_not: float
    move_pct: float
    burst_ratio: float
    imbalance: float
    book_imb: float | None
    spread_bps: float | None
    max_spread: float
    baseline_buy_share: float
    hawkes_ratio: float
    hawkes_excitation: float
    hawkes_slope: float


@dataclass(frozen=True, slots=True)
class MicroThresholds:
    min_trades: int
    min_buy_notional: float
    burst_multiple: float
    min_imbalance: float
    min_move_pct: float
    min_accel: float
    min_delta_divergence: float
    min_book_imb: float
    hawkes_enabled: bool
    hawkes_gate_enabled: bool
    hawkes_min_ratio: float
    hawkes_min_excitation: float
    hawkes_min_slope: float


@dataclass(frozen=True, slots=True)
class MicroCandidate:
    score: float
    accel: float
    delta_divergence: float
    move_pct: float
    burst_ratio: float
    imbalance: float
    book_imb: float
    spread_bps: float
    max_spread: float
    hawkes_ratio: float
    hawkes_excitation: float
    hawkes_slope: float


class MicroIgnitionSignal:
    """Order-book ignition entry signal: cooldowns + tape acceleration + microstructure thresholds + composite score."""

    _ACCEL_RETURN_FLOOR = 5e-4
    _BURST_NORM = 9.0
    _IMBALANCE_NORM = 5.0
    _MOVE_NORM = 0.015
    _ACCEL_NORM = 2.5
    _BOOK_IMB_NORM = 0.8
    _DELTA_NORM = 1.8
    _HAWKES_RATIO_NORM = 2.0
    _HAWKES_EXC_NORM = 4.0
    _WEIGHTS = (3.0, 2.5, 2.4, 1.8, 1.4, 0.7, 0.8, 0.6, -0.7)

    def __init__(
        self,
        *,
        exit_cooldown_sec: DynamicValue,
        candidate_cooldown_sec: DynamicValue,
        cost_floor_multiplier: DynamicValue,
    ) -> None:
        self._cost_floor_multiplier = cost_floor_multiplier
        self._exit_cooldown = Cooldown(exit_cooldown_sec)
        self._candidate_cooldown = Cooldown(candidate_cooldown_sec)
        self._accel = ReturnRatio()
        self._buy_share = Ratio()
        self._delta_div = Ratio()
        self._spread_norm = Ratio()
        self._clamps = tuple(Clamp() for _ in range(9))
        self._blend = WeightedSum()
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_candidate: MicroCandidate | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def arm_exit_cooldown(self, now: float) -> None:
        self._exit_cooldown.arm(now)

    def measure(
        self,
        *,
        now: float,
        snap: MicroSnapshot,
        thresholds: MicroThresholds,
        recent_prices: Sequence[float],
        cost_floor: float,
    ) -> MicroCandidate | None:
        self._exit_cooldown.process(State({"list": [now]}))

        if self._exit_cooldown.readout() < 1.0:
            return None

        self._candidate_cooldown.process(State({"list": [now]}))

        if self._candidate_cooldown.readout() < 1.0:
            return None

        if snap.trades < thresholds.min_trades:
            return None

        if snap.buy_not < thresholds.min_buy_notional:
            return None

        accel = self._accel_from_prices(recent_prices)

        if accel is None:
            return None

        if snap.move_pct < cost_floor * self._cost_floor_multiplier.readout().f[0]:
            return None

        if thresholds.hawkes_enabled and thresholds.hawkes_gate_enabled:
            if (
                snap.hawkes_ratio < thresholds.hawkes_min_ratio
                or snap.hawkes_excitation < thresholds.hawkes_min_excitation
                or snap.hawkes_slope < thresholds.hawkes_min_slope
            ):
                return None

        if snap.book_imb is None or snap.spread_bps is None:
            return None

        flow = snap.buy_not + snap.sell_not
        self._buy_share.process(State({"list": [snap.buy_not, flow, 1.0]}))
        buy_share = self._buy_share.readout()

        self._delta_div.process(State({"list": [buy_share, snap.baseline_buy_share, 0.05]}))
        delta_div = self._delta_div.readout()

        if not self._composite_gate(snap, thresholds, accel, delta_div):
            return None

        self._spread_norm.process(State({"list": [snap.spread_bps, snap.max_spread, 1e-9]}))
        spread_term = self._spread_norm.readout()

        normalized = (
            snap.burst_ratio / self._BURST_NORM,
            snap.imbalance / self._IMBALANCE_NORM,
            snap.move_pct / self._MOVE_NORM,
            accel / self._ACCEL_NORM,
            (snap.book_imb - 1.0) / self._BOOK_IMB_NORM,
            delta_div / self._DELTA_NORM,
            (snap.hawkes_ratio - 1.0) / self._HAWKES_RATIO_NORM,
            snap.hawkes_excitation / self._HAWKES_EXC_NORM,
            spread_term,
        )
        clamped = tuple(self._clamp_unit(c, x) for c, x in zip(self._clamps, normalized))

        self._blend.process(State({
            "weights": list(self._WEIGHTS),
            "values": list(clamped),
        }))
        score = self._blend.readout()

        self._candidate_cooldown.arm(now)

        out = MicroCandidate(
            score=score,
            accel=accel,
            delta_divergence=delta_div,
            move_pct=snap.move_pct,
            burst_ratio=snap.burst_ratio,
            imbalance=snap.imbalance,
            book_imb=snap.book_imb,
            spread_bps=snap.spread_bps,
            max_spread=snap.max_spread,
            hawkes_ratio=snap.hawkes_ratio,
            hawkes_excitation=snap.hawkes_excitation,
            hawkes_slope=snap.hawkes_slope,
        )
        self._last_candidate = out
        return out

    def _accel_from_prices(self, prices: Sequence[float]) -> float | None:
        if len(prices) < 8:
            return None

        p0, p1, p2 = prices[-8], prices[-4], prices[-1]

        if p0 <= 0 or p1 <= 0:
            return None

        if (p2 / p1 - 1.0) <= 0:
            return None

        self._accel.process(State({"list": [p0, p1, p2, self._ACCEL_RETURN_FLOOR]}))
        return self._accel.readout()

    @staticmethod
    def _composite_gate(
        snap: MicroSnapshot,
        t: MicroThresholds,
        accel: float,
        delta_div: float,
    ) -> bool:
        return (
            snap.burst_ratio >= t.burst_multiple
            and snap.imbalance >= t.min_imbalance
            and snap.move_pct >= t.min_move_pct
            and accel >= t.min_accel
            and delta_div >= t.min_delta_divergence
            and snap.book_imb is not None and snap.book_imb >= t.min_book_imb
            and snap.spread_bps is not None and snap.spread_bps <= snap.max_spread
        )

    @staticmethod
    def _clamp_unit(clamp: Clamp, x: float) -> float:
        clamp.process(State({"list": [x, 0.0, 1.0]}))
        return clamp.readout()
