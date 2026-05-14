from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from numeric.blend import Median
from numeric.candle import ClosePosition, RangePercent
from numeric.cooldown import Cooldown
from numeric.dynamic import DynamicValue
from numeric.ratio import Ratio
from numeric.state import State


@dataclass(frozen=True, slots=True)
class Candle:
    """OHLCV bar; matches the (open, high, low, close, volume) tuple shape used by hydra/."""

    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class PumpDetection:
    spike_pct: float
    burst_x: float
    high: float
    close: float
    volume: float


class PumpSignal:
    """Multi-minute pump detector: spike + base-range gate + volume burst + close position + cooldown."""

    _BASE_RANGE_MAX_FRACTION = 0.55

    def __init__(
        self,
        *,
        baseline_mins: int,
        min_spike_pct: DynamicValue,
        vol_burst_x: DynamicValue,
        pump_candle_upper_pct: DynamicValue,
        cooldown_sec: DynamicValue,
    ) -> None:
        self._baseline_mins = baseline_mins
        self._min_spike_pct = min_spike_pct
        self._vol_burst_x = vol_burst_x
        self._pump_candle_upper_pct = pump_candle_upper_pct

        self._spike = RangePercent()
        self._base_range = RangePercent()
        self._burst = Ratio()
        self._close_pos = ClosePosition()
        self._base_vol = Median()
        self._cooldown = Cooldown(cooldown_sec)
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_detection: PumpDetection | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def measure(self, *, now: float, history: Sequence[Candle]) -> PumpDetection | None:
        if len(history) < self._baseline_mins + 1:
            return None

        sample = history[-(self._baseline_mins + 1):]
        baseline = sample[:-1]
        latest = sample[-1]

        self._base_vol.process(State({"list": [c.volume for c in baseline]}))
        base_vol = self._base_vol.readout()

        if base_vol <= 0:
            return None

        min_low = min(c.low for c in baseline)
        max_high = max(c.high for c in baseline)

        if min_low <= 0:
            return None

        self._spike.process(State({"list": [latest.high, latest.low]}))
        spike_pct = self._spike.readout()

        if spike_pct < self._min_spike_pct.readout().f[0]:
            return None

        self._base_range.process(State({"list": [max_high, min_low]}))
        base_range_pct = self._base_range.readout()

        if base_range_pct > spike_pct * self._BASE_RANGE_MAX_FRACTION:
            return None

        self._burst.process(State({"list": [latest.volume, base_vol, 1e-9]}))
        burst_x = self._burst.readout()

        if burst_x < self._vol_burst_x.readout().f[0]:
            return None

        if latest.high > latest.low:
            self._close_pos.process(State({"list": [latest.high, latest.low, latest.close]}))

            if self._close_pos.readout() < self._pump_candle_upper_pct.readout().f[0]:
                return None

        self._cooldown.process(State({"list": [now]}))

        if self._cooldown.readout() < 1.0:
            return None

        self._cooldown.arm(now)

        out = PumpDetection(
            spike_pct=spike_pct,
            burst_x=burst_x,
            high=latest.high,
            close=latest.close,
            volume=latest.volume,
        )
        self._last_detection = out
        return out
