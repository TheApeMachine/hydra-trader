from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from numeric.blend import Median, WeightedSum
from numeric.candle import ClosePosition
from numeric.clamp import Clamp
from numeric.dynamic import DynamicValue
from numeric.peak_watch import PeakWatch
from numeric.ratio import Ratio
from numeric.state import State

from .pump import Candle, PumpDetection


@dataclass(frozen=True, slots=True)
class ReclaimThresholds:
    lookback: int
    peak_update_window_sec: float
    watch_max_age_sec: float
    invalidate_drop: float
    pullback_min: float
    pullback_max: float
    vol_floor: float
    min_close_pos: float
    sweet_drop: float
    sweet_tolerance: float


@dataclass(frozen=True, slots=True)
class ReclaimInputs:
    symbol: str
    history: Sequence[Candle]
    latest: Candle
    prev_close: float
    book_quality_ok: bool


@dataclass(frozen=True, slots=True)
class ReclaimCandidate:
    score: float
    drop: float
    vol_ratio: float
    close_pos: float
    pullback_quality: float


class MacroReclaimSignal:
    """Pullback-reclaim entry: pump anchor seeded externally; confirms on drop band + reclaim candle."""

    _SPIKE_NORM = 28.0
    _BURST_NORM = 18.0
    _VOL_NORM = 3.0
    _WEIGHTS = (2.0, 2.0, 2.5, 2.5, 1.0)

    def __init__(self) -> None:
        self._watch = PeakWatch()
        self._med_vol = Median()
        self._vol_ratio = Ratio()
        self._close_pos = ClosePosition()
        self._pb_clamp = Clamp()
        self._spike_clamp = Clamp()
        self._burst_clamp = Clamp()
        self._vol_clamp = Clamp()
        self._cp_clamp = Clamp()
        self._blend = WeightedSum()
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_candidate: ReclaimCandidate | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def seed_from_pump(self, *, symbol: str, now: float, detection: PumpDetection) -> None:
        self._watch.seed(
            symbol,
            now=now,
            anchor=detection.high,
            payload={"spike_pct": detection.spike_pct, "burst_x": detection.burst_x},
        )

    def forget(self, symbol: str) -> None:
        self._watch.forget(symbol)

    def measure(
        self,
        *,
        now: float,
        inputs: ReclaimInputs,
        thresholds: ReclaimThresholds,
    ) -> ReclaimCandidate | None:
        if inputs.symbol not in self._watch:
            return None

        self._watch.update_anchor(
            inputs.symbol,
            now=now,
            high=inputs.latest.high,
            update_window_sec=thresholds.peak_update_window_sec,
        )

        drop = self._watch.drop(inputs.symbol, close=inputs.latest.close)

        if drop is None:
            return None

        if self._watch.expire_if(
            inputs.symbol,
            now=now,
            max_age_sec=thresholds.watch_max_age_sec,
            drop=drop,
            invalidate_drop=thresholds.invalidate_drop,
        ):
            return None

        if drop < thresholds.pullback_min or drop > thresholds.pullback_max:
            return None

        if len(inputs.history) < thresholds.lookback + 1:
            return None

        recent = inputs.history[-(thresholds.lookback + 1):-1]
        closes = [c.close for c in recent]
        vols = [c.volume for c in recent]

        if not closes:
            return None

        mean_close = sum(closes) / len(closes)
        self._med_vol.process(State({"list": vols}))
        med_vol = self._med_vol.readout()

        if med_vol > 0:
            self._vol_ratio.process(State({"list": [inputs.latest.volume, med_vol, 1e-9]}))
            vol_r = self._vol_ratio.readout()
        else:
            vol_r = 0.0

        self._close_pos.process(State({"list": [inputs.latest.high, inputs.latest.low, inputs.latest.close]}))
        cp = self._close_pos.readout()

        if not self._reclaim_candle_ok(inputs.latest, closes[-1], mean_close, cp, vol_r, thresholds, inputs.book_quality_ok):
            return None

        entry = self._watch.get(inputs.symbol)

        if entry is None:
            return None

        pb_q = self._pullback_quality(drop, thresholds)

        score = self._score(entry.payload, pb_q, vol_r, cp)

        out = ReclaimCandidate(
            score=score,
            drop=drop,
            vol_ratio=vol_r,
            close_pos=cp,
            pullback_quality=pb_q,
        )
        self._last_candidate = out
        return out

    @staticmethod
    def _reclaim_candle_ok(
        latest: Candle,
        prev_close: float,
        mean_close: float,
        cp: float,
        vol_r: float,
        t: ReclaimThresholds,
        book_quality_ok: bool,
    ) -> bool:
        return (
            latest.close > latest.open
            and latest.close > prev_close
            and latest.close > mean_close
            and cp >= t.min_close_pos
            and vol_r >= t.vol_floor
            and book_quality_ok
        )

    def _pullback_quality(self, drop: float, t: ReclaimThresholds) -> float:
        raw = 1.0 - abs(drop - t.sweet_drop) / max(t.sweet_tolerance, 1e-9)
        self._pb_clamp.process(State({"list": [raw, 0.0, 1.0]}))
        return self._pb_clamp.readout()

    def _score(self, payload: dict, pb_q: float, vol_r: float, cp: float) -> float:
        spike_pct = float(payload.get("spike_pct", 0.0))
        burst_x = float(payload.get("burst_x", 0.0))

        self._spike_clamp.process(State({"list": [spike_pct / self._SPIKE_NORM, 0.0, 1.0]}))
        self._burst_clamp.process(State({"list": [burst_x / self._BURST_NORM, 0.0, 1.0]}))
        self._vol_clamp.process(State({"list": [vol_r / self._VOL_NORM, 0.0, 1.0]}))
        self._cp_clamp.process(State({"list": [cp, 0.0, 1.0]}))

        self._blend.process(State({
            "weights": list(self._WEIGHTS),
            "values": [
                self._spike_clamp.readout(),
                self._burst_clamp.readout(),
                pb_q,
                self._vol_clamp.readout(),
                self._cp_clamp.readout(),
            ],
        }))
        return self._blend.readout()
