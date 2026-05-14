from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from numeric.blend import Median, WeightedSum
from numeric.candle import ClosePosition
from numeric.clamp import Clamp
from numeric.dynamic import DynamicValue
from numeric.ratio import Ratio
from numeric.returns import PercentReturn
from numeric.state import State

from .pump import Candle


@dataclass(frozen=True, slots=True)
class ThrustThresholds:
    lookback: int
    min_ret_3m: float
    min_ret_5m: float
    max_ret_5m: float
    min_vol_x: float
    max_vol_x: float
    min_close_pos: float
    min_candle_ret: float
    max_spread_bps: float


@dataclass(frozen=True, slots=True)
class ThrustInputs:
    history: Sequence[Candle]
    latest: Candle
    spread_bps: float | None
    cost_floor: float
    cost_floor_multiplier: float


@dataclass(frozen=True, slots=True)
class ThrustCandidate:
    score: float
    r3: float
    r5: float
    r15: float
    vol_x: float
    close_pos: float
    candle_ret: float


class MacroThrustSignal:
    """Macro momentum thrust entry: multi-minute returns + volume burst + close position + composite score."""

    _R5_NORM = 0.07
    _R3_NORM = 0.035
    _VOL_NORM = 8.0
    _CANDLE_NORM = 0.025
    _MIN_CANDLE_RET = 0.0025
    _WEIGHTS = (3.0, 2.0, 2.5, 1.5, 1.0, -0.8)

    def __init__(self) -> None:
        self._r3 = PercentReturn()
        self._r5 = PercentReturn()
        self._r15 = PercentReturn()
        self._candle_ret = PercentReturn()
        self._vol_burst = Ratio()
        self._med_vol = Median()
        self._close_pos = ClosePosition()
        self._spread_norm = Ratio()
        self._clamps = tuple(Clamp() for _ in range(5))
        self._blend = WeightedSum()
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_candidate: ThrustCandidate | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def measure(self, *, thresholds: ThrustThresholds, inputs: ThrustInputs) -> ThrustCandidate | None:
        hist = inputs.history
        latest = inputs.latest

        if len(hist) < thresholds.lookback + 1:
            return None

        if latest.close <= 0:
            return None

        r3 = self._anchor_return(hist, 4, latest.close)
        r5 = self._anchor_return(hist, 6, latest.close)
        r15 = self._anchor_return(hist, 16, latest.close)

        if r3 < thresholds.min_ret_3m:
            return None

        if r5 < thresholds.min_ret_5m or r5 > thresholds.max_ret_5m:
            return None

        if r15 < 0:
            return None

        baseline = hist[-(thresholds.lookback + 1):-1]
        self._med_vol.process(State({"list": [c.volume for c in baseline]}))
        med_vol = self._med_vol.readout()

        if med_vol <= 0:
            return None

        self._vol_burst.process(State({"list": [latest.volume, med_vol, 1e-9]}))
        vol_x = self._vol_burst.readout()

        if vol_x > thresholds.max_vol_x or vol_x < thresholds.min_vol_x:
            return None

        self._close_pos.process(State({"list": [latest.high, latest.low, latest.close]}))
        cp = self._close_pos.readout()

        if cp < thresholds.min_close_pos or latest.close <= latest.open:
            return None

        self._candle_ret.process(State({"list": [latest.open, latest.close]}))
        candle_ret = self._candle_ret.readout() / 100.0

        if candle_ret < self._MIN_CANDLE_RET or candle_ret < thresholds.min_candle_ret:
            return None

        if abs(r5) < inputs.cost_floor * inputs.cost_floor_multiplier:
            return None

        sp_penalty = self._spread_penalty(inputs.spread_bps, thresholds.max_spread_bps)

        normalized = (
            r5 / self._R5_NORM,
            r3 / self._R3_NORM,
            vol_x / self._VOL_NORM,
            cp,
            candle_ret / self._CANDLE_NORM,
        )
        clamped = tuple(self._clamp_unit(c, x) for c, x in zip(self._clamps, normalized))

        self._blend.process(State({
            "weights": list(self._WEIGHTS),
            "values": [*clamped, sp_penalty],
        }))
        score = self._blend.readout()

        out = ThrustCandidate(
            score=score,
            r3=r3,
            r5=r5,
            r15=r15,
            vol_x=vol_x,
            close_pos=cp,
            candle_ret=candle_ret,
        )
        self._last_candidate = out
        return out

    def _anchor_return(self, hist: Sequence[Candle], offset_from_end: int, close: float) -> float:
        if len(hist) < offset_from_end:
            return 0.0

        anchor = hist[-offset_from_end].close

        if anchor <= 0:
            return 0.0

        self._r3.process(State({"list": [anchor, close]}))
        return self._r3.readout() / 100.0

    def _spread_penalty(self, spread_bps: float | None, max_spread_bps: float) -> float:
        if spread_bps is None:
            return 0.0

        self._spread_norm.process(State({"list": [spread_bps, max(max_spread_bps, 1.0), 1e-9]}))
        return min(1.0, max(0.0, self._spread_norm.readout()))

    @staticmethod
    def _clamp_unit(clamp: Clamp, x: float) -> float:
        clamp.process(State({"list": [x, 0.0, 1.0]}))
        return clamp.readout()
