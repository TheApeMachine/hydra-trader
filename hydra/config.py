from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "hydra_best.json"
RELAXED_PARAMS_PATH = PROJECT_ROOT / "hydra_relaxed.json"
CANDIDATE_PARAMS_PATH = PROJECT_ROOT / "hydra_candidate.json"


def _best_json_validated_zero_trades(data: dict[str, Any]) -> bool:
    """True if Optuna export includes validation rows and every one has trades==0."""
    rec = data.get("best_validation_by_recording")
    if not isinstance(rec, list) or not rec:
        return False
    for item in rec:
        if not isinstance(item, dict):
            return False
        m = item.get("metrics")
        if not isinstance(m, dict):
            return False
        if int(m.get("trades", 0)) > 0:
            return False
    return True


def _should_fallback_from_degenerate_best(path_hint: str | Path | None, target: Path) -> bool:
    return path_hint is None and target.resolve() == DEFAULT_PARAMS_PATH.resolve()


def _try_fallback_relaxed(quiet: bool) -> tuple[Config, str]:
    if not RELAXED_PARAMS_PATH.exists():
        if not quiet:
            print(
                "  [config] promoted best validated with 0 trades; "
                f"{RELAXED_PARAMS_PATH.name} missing — using built-in defaults."
            )
        return Config(), "defaults (0-trade best, no relaxed file)"

    cfg = load_config(RELAXED_PARAMS_PATH)
    if not quiet:
        print(
            f"  [config] promoted best validated with 0 trades — loading "
            f"{RELAXED_PARAMS_PATH.name} instead."
        )
    return cfg, f"relaxed ({RELAXED_PARAMS_PATH.name})"


@dataclasses.dataclass
class Config:
    """Tunable strategy parameters.

    Default numeric gates bias toward setups that clear estimated round-trip
    friction (fees + spread model), rather than shaving marginal moves. Overrides
    still go through JSON (``hydra_best.json`` etc.). Macro tape/L2 coupling defaults
    off so hourly signals are not gated on simultaneous tape bursts.
    """

    # tape window + micro/book_ignition
    tape_window_sec: float = 12.0
    tape_baseline_sec: float = 240.0
    micro_min_trades: int = 6
    micro_min_buy_notional: float = 4_000.0  # loosened — was 45k; still meaningful filter
    micro_burst_multiple: float = 2.8
    micro_min_imbalance: float = 1.55
    micro_min_move_pct: float = 0.0035
    # move must clear this × estimated round-trip friction (spread+fees) to micro-scalp
    micro_vs_cost_floor: float = 0.82
    micro_breakout_eps: float = 0.00035
    micro_cooldown_sec: float = 40.0
    micro_candidate_cooldown_sec: float = 2.0
    micro_accel_threshold: float = 1.2
    micro_delta_divergence: float = 1.08

    # Online marked Hawkes-style tape excitation. This is a small two-sided
    # intensity model (buy/sell self + cross excitation), not a full batch MLE.
    hawkes_enabled: bool = True
    hawkes_micro_gate: bool = True
    hawkes_decay_sec: float = 8.0
    hawkes_impulse_cap: float = 260.0
    hawkes_self_excitation: float = 0.82
    hawkes_cross_excitation: float = 0.12
    hawkes_branching_cap: float = 0.94
    hawkes_prediction_interval_sec: float = 1.0
    hawkes_prediction_horizon_sec: float = 60.0
    hawkes_min_buy_sell_ratio: float = 1.35
    hawkes_min_excitation: float = 8.0
    hawkes_min_slope: float = 0.0
    hawkes_exit_enabled: bool = False
    hawkes_exit_min_age_sec: float = 8.0
    hawkes_exit_sell_buy_ratio: float = 1.18
    hawkes_exit_max_ret: float = 0.012

    # macro pump-pullback-reclaim
    baseline_mins: int = 15
    min_spike_pct: float = 5.0
    vol_burst_x: float = 3.2
    pump_candle_upper_pct: float = 0.65
    cooldown_sec: float = 360.0
    watch_max_age: float = 3600.0
    peak_update_window: float = 360.0
    pullback_min: float = 0.042
    pullback_max: float = 0.175
    invalidate_drop: float = 0.24
    reclaim_lookback: int = 4
    reclaim_vol_floor: float = 1.06

    # thrust
    macro_thrust_enabled: bool = False
    thrust_lookback: int = 20
    thrust_min_ret_3m: float = 0.0065
    thrust_min_ret_5m: float = 0.011
    thrust_max_ret_5m: float = 0.12
    thrust_min_vol_x: float = 1.85
    thrust_min_close_pos: float = 0.62
    thrust_ret_vs_floor: float = 1.18  # |r5| must exceed floor × this (spread+fee model)

    # macro live microstructure confirmation (candidate generation — off by default)
    macro_require_microstructure: bool = False
    macro_min_tape_trades: int = 4
    macro_min_tape_notional: float = 4_500.0
    macro_min_tape_buy_share: float = 0.51
    macro_max_spread_bps: float = 42.0
    macro_min_book_imbalance: float = 0.72
    macro_max_vol_x: float = 80.0

    # wallet / fills
    start_capital: float = 200.0
    fee_pct: float = 0.0026
    slippage: float = 0.0005  # fallback only; L2 book-walk is preferred
    candidate_debounce_sec: float = 0.5
    max_candidate_age: float = 35.0
    book_stale_sec: float = 22.0  # multiplexed feeds: allow slower L2 inter-arrival
    max_portfolio_heat: float = 0.50
    daily_loss_limit: float = 0.115
    vol_target_atr: float = 0.012
    liq_buffer: float = 160_000.0
    min_ticket_usd: float = 8.0

    # Paper-wallet stop orders. These simulate long stop-market orders in the
    # wallet/backtester; they are not exchange-native orders.
    paper_stop_orders_enabled: bool = True
    paper_stop_gap_fill: bool = True

    # risk: book_ignition
    bi_hard_stop: float = 0.028
    bi_trail_activate: float = 0.018
    bi_trail_pct: float = 0.012
    bi_take_profit: float = 0.024
    bi_max_hold: float = 480.0
    bi_stall_check: float = 90.0
    bi_stall_ret: float = 0.006

    # risk: macro_thrust
    mt_hard_stop: float = 0.030
    mt_trail_activate: float = 0.018
    mt_trail_pct: float = 0.012
    mt_take_profit: float = 0.026
    mt_max_hold: float = 900.0
    mt_stall_check: float = 240.0
    mt_stall_ret: float = 0.004

    # risk: macro_reclaim_v2
    mr_hard_stop: float = 0.034
    mr_trail_activate: float = 0.020
    mr_trail_pct: float = 0.014
    mr_take_profit: float = 0.030
    mr_max_hold: float = 1200.0
    mr_stall_check: float = 300.0
    mr_stall_ret: float = 0.005

    # early failure cuts for macro entries
    macro_early_fail_sec: float = 120.0
    macro_early_fail_ret: float = -0.012
    macro_no_followthrough_sec: float = 300.0
    macro_no_followthrough_ret: float = 0.004

    # time-efficiency exits: require trades to make progress quickly enough
    time_efficiency_check_sec: float = 210.0
    time_efficiency_min_ret: float = 0.0025
    min_return_velocity_per_min: float = 0.00035
    horizon_trail_tighten_sec: float = 360.0
    horizon_tight_trail_pct: float = 0.008

    # adaptive runtime — microstructure/Tape/BTC cues + rolling returns modulate friction gates (on by default)
    adaptive_enabled: bool = True
    adaptive_refresh_sec: float = 25.0
    adaptive_perf_window_sec: float = 14400.0
    adaptive_min_trades_in_window: int = 4
    adaptive_smooth_alpha: float = 0.16
    adaptive_relax_max_step: float = 0.065
    adaptive_stress_damping: float = 0.55
    adaptive_micro_floor_span: float = 0.10
    adaptive_thrust_floor_span: float = 0.08
    adaptive_score_size_span: float = 0.055
    adaptive_micro_floor_hard_min: float = 0.56
    adaptive_micro_floor_hard_max: float = 0.96
    adaptive_thrust_floor_hard_min: float = 1.02
    adaptive_thrust_floor_hard_max: float = 1.26
    adaptive_score_size_hard_min: float = 0.90
    adaptive_score_size_hard_max: float = 1.085
    adaptive_micro_sample_syms: int = 14
    adaptive_opp_weight_spread: float = 0.28
    adaptive_opp_weight_tape: float = 0.24
    adaptive_opp_weight_flow: float = 0.20
    adaptive_opp_weight_btc: float = 0.28
    adaptive_stress_underperf_den: float = 0.018
    adaptive_stress_loss_frac_ref: float = 0.62

    def risk(self, regime: str) -> dict[str, float]:
        if regime == "book_ignition":
            return {
                "hard_stop": self.bi_hard_stop,
                "trail_activate": self.bi_trail_activate,
                "trail_pct": self.bi_trail_pct,
                "take_profit": self.bi_take_profit,
                "max_hold": self.bi_max_hold,
                "stall_check": self.bi_stall_check,
                "stall_ret": self.bi_stall_ret,
            }
    
        if regime == "macro_thrust":
            return {
                "hard_stop": self.mt_hard_stop,
                "trail_activate": self.mt_trail_activate,
                "trail_pct": self.mt_trail_pct,
                "take_profit": self.mt_take_profit,
                "max_hold": self.mt_max_hold,
                "stall_check": self.mt_stall_check,
                "stall_ret": self.mt_stall_ret,
            }
    
        if regime == "macro_reclaim_v2":
            return {
                "hard_stop": self.mr_hard_stop,
                "trail_activate": self.mr_trail_activate,
                "trail_pct": self.mr_trail_pct,
                "take_profit": self.mr_take_profit,
                "max_hold": self.mr_max_hold,
                "stall_check": self.mr_stall_check,
                "stall_ret": self.mr_stall_ret,
            }
    
        raise ValueError(regime)


def apply_params(cfg: Config, params: dict[str, Any] | None) -> Config:
    fields = {f.name: f.type for f in dataclasses.fields(cfg)}
    params = params or {}

    # Compatibility for tuned files created before the explicit thrust switch.
    # If a params file tuned thrust-specific fields, preserve that behavior
    # unless it opts out with macro_thrust_enabled=false.
    if "macro_thrust_enabled" not in params and any(
        key.startswith("thrust_") or key.startswith("mt_")
        for key in params
    ):
        cfg.macro_thrust_enabled = True
    
    for key, value in params.items():
        if key not in fields:
            continue
    
        try:
            typ = fields[key]
            if typ is int or typ == "int":
                value = int(value)
            elif typ is float or typ == "float":
                value = float(value)
            elif typ is bool or typ == "bool":
                value = bool(value)
        except (TypeError, ValueError):
            pass
    
        setattr(cfg, key, value)
    
    return cfg


def load_config(path: str | Path, base: Config | None = None) -> Config:
    cfg = dataclasses.replace(base) if base is not None else Config()

    with Path(path).open() as f:
        data = json.load(f)

    return apply_params(
        cfg, 
        data.get("best_params") if isinstance(data, dict) and "best_params" in data else data
    )


def maybe_load_config(
    path: str | Path | None = None, 
    no_params: bool = False, 
    quiet: bool = False
) -> tuple[Config, str]:
    if no_params:
        return Config(), "defaults"

    target = Path(path) if path else DEFAULT_PARAMS_PATH

    if target.exists():
        if _should_fallback_from_degenerate_best(path, target):
            try:
                raw = json.loads(target.read_text())
            except (OSError, json.JSONDecodeError):
                raw = None
            if isinstance(raw, dict) and _best_json_validated_zero_trades(raw):
                return _try_fallback_relaxed(quiet)

        cfg = load_config(target)
        source = f"tuned ({target.name})"

        if not quiet:
            print(f"Loaded tuned params from {target.resolve()}")
            print(
                "  Hint: pass `--no-params` to skip JSON and use the repo’s exploratory built-in gates."
            )

        return cfg, source

    return Config(), "defaults"
