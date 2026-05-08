from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "hydra_best.json"
CANDIDATE_PARAMS_PATH = PROJECT_ROOT / "hydra_candidate.json"


@dataclasses.dataclass
class Config:
    """Tunable strategy parameters.

    These defaults are intentionally more defensive than the early prototype:
    macro entries must move quickly, risk is capped, and live macro signals
    require fresh L2/tape confirmation in the strategy layer.
    """

    # tape window + micro/book_ignition
    tape_window_sec: float = 12.0
    tape_baseline_sec: float = 240.0
    micro_min_trades: int = 12
    micro_min_buy_notional: float = 45_000.0
    micro_burst_multiple: float = 4.5
    micro_min_imbalance: float = 2.2
    micro_min_move_pct: float = 0.005
    micro_breakout_eps: float = 0.00035
    micro_cooldown_sec: float = 75.0
    micro_candidate_cooldown_sec: float = 3.5
    micro_accel_threshold: float = 1.75
    micro_delta_divergence: float = 1.22

    # macro pump-pullback-reclaim
    baseline_mins: int = 20
    min_spike_pct: float = 7.0
    vol_burst_x: float = 4.8
    pump_candle_upper_pct: float = 0.70
    cooldown_sec: float = 360.0
    watch_max_age: float = 3600.0
    peak_update_window: float = 360.0
    pullback_min: float = 0.042
    pullback_max: float = 0.175
    invalidate_drop: float = 0.24
    reclaim_lookback: int = 4
    reclaim_vol_floor: float = 1.18

    # thrust
    thrust_lookback: int = 20
    thrust_min_ret_3m: float = 0.011
    thrust_min_ret_5m: float = 0.019
    thrust_max_ret_5m: float = 0.11
    thrust_min_vol_x: float = 2.8
    thrust_min_close_pos: float = 0.76

    # macro live microstructure confirmation
    macro_require_microstructure: bool = True
    macro_min_tape_trades: int = 6
    macro_min_tape_notional: float = 15_000.0
    macro_min_tape_buy_share: float = 0.54
    macro_max_spread_bps: float = 18.0
    macro_min_book_imbalance: float = 0.85
    macro_max_vol_x: float = 60.0

    # wallet / fills
    start_capital: float = 200.0
    fee_pct: float = 0.0026
    slippage: float = 0.0005  # fallback only; L2 book-walk is preferred
    candidate_debounce_sec: float = 0.75
    max_candidate_age: float = 8.0
    max_portfolio_heat: float = 0.50
    daily_loss_limit: float = 0.115
    vol_target_atr: float = 0.012
    liq_buffer: float = 160_000.0
    min_ticket_usd: float = 8.0

    # risk: book_ignition
    bi_hard_stop: float = 0.028
    bi_trail_activate: float = 0.018
    bi_trail_pct: float = 0.012
    bi_take_profit: float = 0.024
    bi_max_hold: float = 720.0
    bi_stall_check: float = 120.0
    bi_stall_ret: float = 0.005

    # risk: macro_thrust
    mt_hard_stop: float = 0.030
    mt_trail_activate: float = 0.018
    mt_trail_pct: float = 0.012
    mt_take_profit: float = 0.026
    mt_max_hold: float = 1800.0
    mt_stall_check: float = 420.0
    mt_stall_ret: float = 0.003

    # risk: macro_reclaim_v2
    mr_hard_stop: float = 0.034
    mr_trail_activate: float = 0.020
    mr_trail_pct: float = 0.014
    mr_take_profit: float = 0.030
    mr_max_hold: float = 2400.0
    mr_stall_check: float = 540.0
    mr_stall_ret: float = 0.004

    # early failure cuts for macro entries
    macro_early_fail_sec: float = 180.0
    macro_early_fail_ret: float = -0.012
    macro_no_followthrough_sec: float = 420.0
    macro_no_followthrough_ret: float = 0.003

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
    
    for key, value in (params or {}).items():
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
        cfg = load_config(target)
        source = f"tuned ({target.name})"

        if not quiet:
            print(f"Loaded tuned params from {target.resolve()}")

        return cfg, source

    return Config(), "defaults"
