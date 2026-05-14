"""Moment-aware modulation of friction-style gates — opportunity + stress blended with safeguards.

Relaxation is normalized to approximately [-1, 1]:
  + relaxed  → somewhat easier entries (micro / thrust floors move with the market),
               slightly larger score-based sizing.
  − defensive → tighter entries when rolling performance is weak.

Safeguards: EMA smoothing, max delta per refresh, absolute clamps per parameter,
respect for min-trade sample counts before stressing PnL, and refresh cadence.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .config import Config
from .constants import BTC_SYMBOL
from .market import MarketStore
from .utils import clamp, median


@dataclass
class AdaptiveSnapshot:
    """Last refresh diagnostics (telemetry / dashboards)."""

    relax: float
    opportunity_raw: float
    stress_raw: float
    n_trades_window: int
    samples_ok: int


class AdaptiveContext:
    """Rolling performance + instantaneous market cues → bounded gate adjustments."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._relax = 0.0
        self._last_refresh_ts = -1e30
        self._closes = deque[tuple[float, float]](maxlen=512)
        self.last_snap: AdaptiveSnapshot | None = None

    # ── bookkeeping ─────────────────────────────────────
    def note_close(self, ts: float, return_decimal: float) -> None:
        self._closes.append((float(ts), float(return_decimal)))

    def maybe_refresh(
        self,
        now: float,
        market: MarketStore,
        trade_symbols: Iterable[str],
        force: bool = False,
    ) -> AdaptiveSnapshot | None:
        cfg = self.cfg
        if not cfg.adaptive_enabled:
            self.last_snap = None
            return None

        if not force and (now - self._last_refresh_ts) < cfg.adaptive_refresh_sec:
            return self.last_snap

        self._last_refresh_ts = now

        opp = self._compute_opportunity(now, market, trade_symbols)
        stress_raw = self._compute_stress(now)
        opp_w = opp * (1.0 - cfg.adaptive_stress_damping * stress_raw)
        tgt = clamp(opp_w - stress_raw, -1.0, 1.0)

        prev = self._relax
        smoothed = prev + cfg.adaptive_smooth_alpha * (tgt - prev)
        stepped = clamp(
            smoothed,
            prev - cfg.adaptive_relax_max_step,
            prev + cfg.adaptive_relax_max_step,
        )
        self._relax = clamp(stepped, -1.0, 1.0)

        n_recent = sum(1 for t, _ in self._closes if t >= now - cfg.adaptive_perf_window_sec)

        snap = AdaptiveSnapshot(
            relax=self._relax,
            opportunity_raw=float(opp),
            stress_raw=float(stress_raw),
            n_trades_window=n_recent,
            samples_ok=self._samples_effective(market, trade_symbols),
        )
        self.last_snap = snap
        return snap

    # ── effective knobs (consume base cfg values as anchors) ─
    def effective_micro_vs_cost_floor(self, base_floor: float) -> float:
        if not self.cfg.adaptive_enabled:
            return base_floor

        scaled = base_floor * (1.0 - self.cfg.adaptive_micro_floor_span * self._relax)
        return clamp(
            scaled,
            self.cfg.adaptive_micro_floor_hard_min,
            self.cfg.adaptive_micro_floor_hard_max,
        )

    def effective_thrust_ret_vs_floor(self, base_floor: float) -> float:
        if not self.cfg.adaptive_enabled:
            return base_floor

        scaled = base_floor * (1.0 - self.cfg.adaptive_thrust_floor_span * self._relax)
        return clamp(
            scaled,
            self.cfg.adaptive_thrust_floor_hard_min,
            self.cfg.adaptive_thrust_floor_hard_max,
        )

    def score_size_boost(self) -> float:
        if not self.cfg.adaptive_enabled:
            return 1.0

        return clamp(
            1.0 + self.cfg.adaptive_score_size_span * self._relax,
            self.cfg.adaptive_score_size_hard_min,
            self.cfg.adaptive_score_size_hard_max,
        )

    # ── internals ────────────────────────────────────────
    def _samples_effective(self, market: MarketStore, symbols: Iterable[str]) -> int:
        n_ok = 0
        syms = list(symbols)
        lim = max(1, min(int(self.cfg.adaptive_micro_sample_syms), len(syms) or 1))
        for sym in syms[:lim]:
            sn = market.snapshot_micro(sym)
            if sn is not None and int(sn.get("trades") or 0) >= max(5, self.cfg.micro_min_trades // 2):
                n_ok += 1

        return n_ok

    def _compute_opportunity(self, _now: float, market: MarketStore, symbols: Iterable[str]) -> float:
        syms = list(symbols)
        if not syms:
            return 0.48

        lim = max(1, min(int(self.cfg.adaptive_micro_sample_syms), len(syms)))
        spreads: list[float] = []
        bursts: list[float] = []
        directs: list[float] = []

        for sym in syms[:lim]:
            sn = market.snapshot_micro(sym)
            if sn is None or int(sn.get("trades") or 0) < 5:
                continue
            sb = sn.get("spread_bps")
            if sb is not None and sb > 0:
                spreads.append(float(sb))

            br_raw = float(sn.get("burst_ratio") or 0.0)
            bursts.append(clamp(br_raw / 12.0, 0.0, 1.0))
            signed = float(sn.get("flow_signed_usd_per_s") or 0.0)
            directs.append(1.0 if signed >= 80.0 else (0.7 if signed >= 15.0 else 0.4))

        ref_spread = max(market.estimated_cost_floor(BTC_SYMBOL) * 9500.0, 42.0)
        if spreads:
            med_spread = median(spreads)
            tight = clamp((ref_spread - med_spread) / ref_spread, 0.0, 1.0)
        else:
            tight = 0.52

        if bursts:
            br = float(median(bursts))
        else:
            br = 0.42

        if directs:
            flow_q = clamp(sum(directs) / len(directs), 0.0, 1.0)
        else:
            flow_q = 0.45

        btc_ok = 1.0 if market.btc_context_ok(strict=False) else (0.18 if market.btc_flush_active() else 0.41)
        btc_w = clamp(btc_ok, 0.0, 1.0)

        samp = spreads or bursts or directs
        sparsity = 0.14 if len(samp) < 4 else (0.10 if len(samp) < 6 else 0.0)

        wc = self.cfg
        raw = (
            wc.adaptive_opp_weight_spread * tight
            + wc.adaptive_opp_weight_tape * br
            + wc.adaptive_opp_weight_flow * flow_q
            + wc.adaptive_opp_weight_btc * btc_w
            - sparsity
        )
        return clamp(raw, 0.0, 1.0)

    def _compute_stress(self, now: float) -> float:
        cfg = self.cfg
        cutoff = now - cfg.adaptive_perf_window_sec
        hits = [(t, r) for t, r in self._closes if t >= cutoff]
        n = len(hits)
        if n < cfg.adaptive_min_trades_in_window:
            return 0.0

        mean_r = sum(r for _, r in hits) / n
        frac_loss = sum(1.0 for _, r in hits if r < 0.0) / n

        # Average loss hurts; losing streak dominance hurts.
        underperf = clamp(-mean_r / max(cfg.adaptive_stress_underperf_den, 1e-9), 0.0, 1.0)
        chop = clamp(frac_loss / max(cfg.adaptive_stress_loss_frac_ref, 0.01), 0.0, 1.0)

        return clamp(0.60 * underperf + 0.55 * chop, 0.0, 1.0)
