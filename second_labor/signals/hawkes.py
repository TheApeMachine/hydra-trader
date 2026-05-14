from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from ..numeric.dynamic import DynamicValue
from ..numeric.state import State
from ..numeric.affine import IntensityAffine
from ..numeric.excitation import DecayedExcitation
from ..numeric.event import PostEventJump, ExpectedEvents


class Publisher(Protocol):
    def publish(self, record: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class ForwardPrediction:
    expected_events: float
    horizon_sec: float
    intensity_now: float
    evaluated_at: float


class HawkesSignal:
    """Self-exciting trade/event intensity composed from numeric primitives."""

    def __init__(
        self,
        focus_pair_key: str | None = None,
        *,
        mu: DynamicValue,
        alpha: DynamicValue,
        beta: DynamicValue,
        publisher: Publisher | None = None,
    ) -> None:
        self.focus_pair_key = focus_pair_key
        self._publisher = publisher
        self._r = DynamicValue(initial=State({"list": [0.0]}))
        self._t_last = DynamicValue(initial=State({"list": []}))
        self._decay = DecayedExcitation(beta)
        self._intensity = IntensityAffine(mu, alpha)
        self._jump = PostEventJump()
        self._forecast = ExpectedEvents(mu, alpha, beta)
        self._confidence = DynamicValue(initial=State({"list": [0.0]}))
        self._last_prediction: ForwardPrediction | None = None

    def confidence(self) -> float:
        return self._confidence.readout().f[0]

    def measure(self, data: Any | None = None) -> float | None:
        if data is None:
            return None

        t, pair = self._parse_event(data)

        if self.focus_pair_key is not None and pair != self.focus_pair_key:
            return None

        t_last_list = self._t_last.readout().f

        if t_last_list and t < t_last_list[0]:
            raise ValueError("events must be sorted by non-decreasing time")

        r = self._r.readout().f[0]

        self._decay.set_t_last(t_last_list[0] if t_last_list else None)
        self._decay.process(State({"list": [r, t]}))
        r_d = self._decay.readout()

        self._intensity.process(State({"list": [r_d]}))
        lam_before = self._intensity.readout()

        self._jump.process(State({"list": [r_d]}))
        self._r.update({"list": [self._jump.readout()]})
        self._t_last.update({"list": [t]})

        if self._publisher is not None:
            self._publisher.publish({
                "kind": "hawkes",
                "symbol": pair,
                "t": t,
                "intensity": lam_before,
            })

        return lam_before

    def predict_forward(self, horizon_sec: float, *, now: float | None = None) -> ForwardPrediction:
        t_last_list = self._t_last.readout().f

        if now is None:
            now = t_last_list[0] if t_last_list else 0.0

        r = self._r.readout().f[0]

        self._decay.set_t_last(t_last_list[0] if t_last_list else None)
        self._decay.process(State({"list": [r, now]}))
        r_now = self._decay.readout()

        self._intensity.process(State({"list": [r_now]}))
        lam_now = self._intensity.readout()

        self._forecast.process(State({"list": [r_now, horizon_sec]}))
        mean_events = self._forecast.readout()

        pred = ForwardPrediction(
            expected_events=mean_events,
            horizon_sec=horizon_sec,
            intensity_now=lam_now,
            evaluated_at=now,
        )
        self._last_prediction = pred

        if self._publisher is not None:
            self._publisher.publish({
                "kind": "hawkes_forecast",
                "expected_events": mean_events,
                "horizon_sec": horizon_sec,
                "intensity_now": lam_now,
                "evaluated_at": now,
            })

        return pred

    def truth(self, observed_event_count: int) -> float:
        pred = self._last_prediction

        if pred is None or observed_event_count < 0:
            self._confidence.update({"list": [0.0]})
            return 0.0

        lam = max(pred.expected_events, 1e-12)
        err = abs(float(observed_event_count) - lam) / math.sqrt(lam)
        c = float(math.exp(-err))
        self._confidence.update({"list": [c]})

        return c

    @staticmethod
    def _parse_event(data: Any) -> tuple[float, str | None]:
        if isinstance(data, (int, float)):
            return float(data), None

        if isinstance(data, dict):
            t = data.get("t") or data.get("trade.timestamp")

            if t is None:
                raise ValueError("dict events require 't' or 'trade.timestamp'")

            raw_pair = data.get("pair") or data.get("wsname") or data.get("trade.symbol")
            pair = raw_pair if isinstance(raw_pair, str) else None

            return float(t), pair

        raise TypeError(f"unsupported event payload type: {type(data)!r}")
