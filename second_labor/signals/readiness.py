from __future__ import annotations

from dataclasses import dataclass

from numeric.clamp import Clamp, LogClamped
from numeric.dynamic import DynamicValue
from numeric.state import State


@dataclass(frozen=True, slots=True)
class ReadinessInputs:
    """Per-symbol features feeding the readiness score."""

    candidate_score: float
    hawkes_excitation: float
    hawkes_ratio: float
    btc_flush: bool
    btc_ok: bool


@dataclass(frozen=True, slots=True)
class ReadinessReadout:
    score: float
    candidate_contrib: float
    hawkes_dir_contrib: float
    hawkes_mag_contrib: float
    btc_nudge: float


class ReadinessSignal:
    """Per-symbol Buy/Sell readiness in [-100, +100] composed from atomic numerics."""

    _CANDIDATE_SCALE = 50.0 / 10.0
    _CANDIDATE_CAP = 50.0
    _DIR_WEIGHT = 18.0
    _MAG_WEIGHT = 12.0
    _DIR_LO = -2.0
    _DIR_HI = 2.0
    _SCORE_LO = -100.0
    _SCORE_HI = 100.0
    _BTC_FLUSH_NUDGE = -25.0
    _BTC_NOT_OK_NUDGE = -8.0

    def __init__(self) -> None:
        self._cand_clamp = Clamp()
        self._direction = LogClamped()
        self._score_clamp = Clamp()
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_readout: ReadinessReadout | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def measure(self, inputs: ReadinessInputs) -> ReadinessReadout:
        self._cand_clamp.process(State({
            "list": [inputs.candidate_score * self._CANDIDATE_SCALE, 0.0, self._CANDIDATE_CAP],
        }))
        cand_contrib = self._cand_clamp.readout()

        self._direction.process(State({
            "list": [inputs.hawkes_ratio, self._DIR_LO, self._DIR_HI, 1e-9],
        }))
        direction = self._direction.readout()

        excess = max(0.0, inputs.hawkes_excitation - 1.0)
        sign = 1.0 if direction > 0 else -1.0 if direction < 0 else 0.0
        mag_contrib = excess * self._MAG_WEIGHT * sign
        dir_contrib = direction * self._DIR_WEIGHT

        btc_nudge = self._btc_nudge(inputs)

        raw = cand_contrib + dir_contrib + mag_contrib + btc_nudge
        self._score_clamp.process(State({"list": [raw, self._SCORE_LO, self._SCORE_HI]}))

        out = ReadinessReadout(
            score=self._score_clamp.readout(),
            candidate_contrib=cand_contrib,
            hawkes_dir_contrib=dir_contrib,
            hawkes_mag_contrib=mag_contrib,
            btc_nudge=btc_nudge,
        )
        self._last_readout = out
        return out

    @classmethod
    def _btc_nudge(cls, inputs: ReadinessInputs) -> float:
        if inputs.btc_flush:
            return cls._BTC_FLUSH_NUDGE

        if not inputs.btc_ok:
            return cls._BTC_NOT_OK_NUDGE

        return 0.0
