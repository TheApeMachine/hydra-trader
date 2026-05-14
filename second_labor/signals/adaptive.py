from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from numeric.affine_scale import ScaledClamp
from numeric.blend import Mean, Median, WeightedSum
from numeric.clamp import Clamp
from numeric.dynamic import DynamicValue
from numeric.state import State
from numeric.window import TimeWindow


@dataclass(frozen=True, slots=True)
class SymbolMicro:
    """Per-symbol market snapshot feeding opportunity scoring."""

    trades: int
    spread_bps: float | None
    burst_ratio: float
    flow_signed_usd_per_s: float


@dataclass(frozen=True, slots=True)
class OpportunityInputs:
    symbols: Sequence[SymbolMicro]
    ref_spread_bps: float
    btc_ok: bool
    btc_flush: bool
    weight_spread: float
    weight_tape: float
    weight_flow: float
    weight_btc: float


@dataclass(frozen=True, slots=True)
class StressInputs:
    """Windowed trade returns; supply via AdaptiveSignal.note_close."""

    returns: Sequence[float]
    underperf_den: float
    loss_frac_ref: float
    min_trades: int


@dataclass(frozen=True, slots=True)
class AdaptiveReadout:
    relax: float
    opportunity: float
    stress: float
    n_trades_window: int


class OpportunitySignal:
    """Market-side opportunity in [0, 1] composed from spread tightness, tape burst, flow, BTC backdrop."""

    _MIN_TRADES = 5
    _BURST_NORM = 12.0
    _FLOW_STRONG_USD = 80.0
    _FLOW_MILD_USD = 15.0
    _FLOW_STRONG = 1.0
    _FLOW_MILD = 0.7
    _FLOW_WEAK = 0.4
    _DEFAULT_TIGHT = 0.52
    _DEFAULT_BURST = 0.42
    _DEFAULT_FLOW = 0.45
    _BTC_OK = 1.0
    _BTC_FLUSH = 0.18
    _BTC_DEGRADED = 0.41
    _SPARSITY_LOW = 0.14
    _SPARSITY_MED = 0.10

    def __init__(self) -> None:
        self._burst_clamp = Clamp()
        self._tight_clamp = Clamp()
        self._flow_clamp = Clamp()
        self._btc_clamp = Clamp()
        self._final_clamp = Clamp()
        self._burst_median = Median()
        self._flow_mean = Mean()
        self._blend = WeightedSum()
        self._out: float | None = None

    def measure(self, inputs: OpportunityInputs) -> float:
        spreads: list[float] = []
        bursts: list[float] = []
        directs: list[float] = []

        for s in inputs.symbols:
            if s.trades < self._MIN_TRADES:
                continue

            if s.spread_bps is not None and s.spread_bps > 0:
                spreads.append(float(s.spread_bps))

            self._burst_clamp.process(State({"list": [s.burst_ratio / self._BURST_NORM, 0.0, 1.0]}))
            bursts.append(self._burst_clamp.readout())
            directs.append(self._flow_bucket(s.flow_signed_usd_per_s))

        tight = self._tight(spreads, inputs.ref_spread_bps)
        burst = self._median_or_default(bursts, self._DEFAULT_BURST)
        flow = self._mean_clamped(directs, self._DEFAULT_FLOW)
        btc = self._btc_weight(inputs.btc_ok, inputs.btc_flush)

        n_samp = len(spreads or bursts or directs)
        sparsity = self._sparsity_penalty(n_samp)

        self._blend.process(State({
            "weights": [inputs.weight_spread, inputs.weight_tape, inputs.weight_flow, inputs.weight_btc],
            "values": [tight, burst, flow, btc],
        }))
        raw = self._blend.readout() - sparsity

        self._final_clamp.process(State({"list": [raw, 0.0, 1.0]}))
        self._out = self._final_clamp.readout()
        return self._out

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("OpportunitySignal.readout before measure")

        return self._out

    def _flow_bucket(self, signed: float) -> float:
        if signed >= self._FLOW_STRONG_USD:
            return self._FLOW_STRONG

        if signed >= self._FLOW_MILD_USD:
            return self._FLOW_MILD

        return self._FLOW_WEAK

    def _tight(self, spreads: list[float], ref_spread_bps: float) -> float:
        if not spreads:
            return self._DEFAULT_TIGHT

        self._burst_median.process(State({"list": spreads}))
        med = self._burst_median.readout()
        self._tight_clamp.process(State({
            "list": [(ref_spread_bps - med) / max(ref_spread_bps, 1e-9), 0.0, 1.0],
        }))
        return self._tight_clamp.readout()

    def _median_or_default(self, xs: list[float], default: float) -> float:
        if not xs:
            return default

        self._burst_median.process(State({"list": xs}))
        return self._burst_median.readout()

    def _mean_clamped(self, xs: list[float], default: float) -> float:
        if not xs:
            return default

        self._flow_mean.process(State({"list": xs}))
        self._flow_clamp.process(State({"list": [self._flow_mean.readout(), 0.0, 1.0]}))
        return self._flow_clamp.readout()

    def _btc_weight(self, btc_ok: bool, btc_flush: bool) -> float:
        if btc_ok:
            raw = self._BTC_OK
        elif btc_flush:
            raw = self._BTC_FLUSH
        else:
            raw = self._BTC_DEGRADED

        self._btc_clamp.process(State({"list": [raw, 0.0, 1.0]}))
        return self._btc_clamp.readout()

    @classmethod
    def _sparsity_penalty(cls, n: int) -> float:
        if n < 4:
            return cls._SPARSITY_LOW

        if n < 6:
            return cls._SPARSITY_MED

        return 0.0


class StressSignal:
    """Performance-side stress in [0, 1] from windowed trade returns (underperformance + loss-fraction)."""

    _UNDERPERF_WEIGHT = 0.60
    _CHOP_WEIGHT = 0.55

    def __init__(self) -> None:
        self._underperf_clamp = Clamp()
        self._chop_clamp = Clamp()
        self._final_clamp = Clamp()
        self._blend = WeightedSum()
        self._out: float | None = None

    def measure(self, inputs: StressInputs) -> float:
        rets = inputs.returns
        n = len(rets)

        if n < inputs.min_trades:
            self._out = 0.0
            return self._out

        mean_r = sum(rets) / n
        frac_loss = sum(1.0 for r in rets if r < 0.0) / n

        self._underperf_clamp.process(State({
            "list": [-mean_r / max(inputs.underperf_den, 1e-9), 0.0, 1.0],
        }))
        self._chop_clamp.process(State({
            "list": [frac_loss / max(inputs.loss_frac_ref, 0.01), 0.0, 1.0],
        }))

        self._blend.process(State({
            "weights": [self._UNDERPERF_WEIGHT, self._CHOP_WEIGHT],
            "values": [self._underperf_clamp.readout(), self._chop_clamp.readout()],
        }))
        self._final_clamp.process(State({"list": [self._blend.readout(), 0.0, 1.0]}))
        self._out = self._final_clamp.readout()
        return self._out

    def readout(self) -> float:
        if self._out is None:
            raise RuntimeError("StressSignal.readout before measure")

        return self._out


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
    stress_damping: float
    smooth_alpha: float
    relax_max_step: float
    perf_window_sec: float


class AdaptiveSignal:
    """Blend opportunity + stress with EMA smoothing and per-refresh step-limit; output relax ∈ [-1, 1]."""

    def __init__(self, *, history: int = 512) -> None:
        self._opportunity = OpportunitySignal()
        self._stress = StressSignal()
        self._closes = TimeWindow(maxlen=history)
        self._tgt_clamp = Clamp()
        self._step_clamp = Clamp()
        self._abs_clamp = Clamp()
        self._relax = DynamicValue(initial=State({"list": [0.0]}))
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_readout: AdaptiveReadout | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def relax(self) -> float:
        return self._relax.readout().f[0]

    def note_close(self, ts: float, return_decimal: float) -> None:
        self._closes.push(ts, return_decimal)

    def measure(
        self,
        *,
        now: float,
        cfg: AdaptiveConfig,
        opportunity_inputs: OpportunityInputs,
        stress_underperf_den: float,
        stress_loss_frac_ref: float,
        stress_min_trades: int,
    ) -> AdaptiveReadout:
        returns = self._closes.values_within(now, cfg.perf_window_sec)
        opp = self._opportunity.measure(opportunity_inputs)
        stress = self._stress.measure(StressInputs(
            returns=returns,
            underperf_den=stress_underperf_den,
            loss_frac_ref=stress_loss_frac_ref,
            min_trades=stress_min_trades,
        ))

        opp_w = opp * (1.0 - cfg.stress_damping * stress)
        self._tgt_clamp.process(State({"list": [opp_w - stress, -1.0, 1.0]}))
        tgt = self._tgt_clamp.readout()

        prev = self.relax()
        smoothed = prev + cfg.smooth_alpha * (tgt - prev)

        self._step_clamp.process(State({
            "list": [smoothed, prev - cfg.relax_max_step, prev + cfg.relax_max_step],
        }))
        stepped = self._step_clamp.readout()

        self._abs_clamp.process(State({"list": [stepped, -1.0, 1.0]}))
        new_relax = self._abs_clamp.readout()
        self._relax.update({"list": [new_relax]})

        out = AdaptiveReadout(
            relax=new_relax,
            opportunity=opp,
            stress=stress,
            n_trades_window=len(returns),
        )
        self._last_readout = out
        return out


def make_knob() -> ScaledClamp:
    """Convenience: a per-knob `ScaledClamp` for `base * (1 - span*relax)` clamped to [lo, hi]."""
    return ScaledClamp()
