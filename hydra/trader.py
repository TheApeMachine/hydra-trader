from __future__ import annotations

import csv
import dataclasses
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

import numpy as np

from .config import Config
from .constants import BTC_SYMBOL
from .market import MarketStore
from .utils import candle_close_pos_dict, clamp, close_position, iso_from_ts, median


@dataclasses.dataclass
class TraderHooks:
    candidate_expired: Callable[[int], None] = lambda n: None
    candidate_rejected: Callable[[int], None] = lambda n: None
    fill: Callable[[], None] = lambda: None


class HydraTrader:
    """Strategy + paper wallet.

    The trader depends on Config, MarketStore, and Clock. It does not own live
    transport, dashboard state, or Optuna state. Backtests simply instantiate
    another HydraTrader with another MarketStore.
    """

    def __init__(
        self,
        cfg: Config,
        market: MarketStore,
        clock,
        name: str = "Hydra",
        csv_enabled: bool = True,
        verbose: bool = True,
        csv_path: str | None = None,
        hooks: TraderHooks | None = None,
    ):
        self.cfg = cfg
        self.market = market
        self.clock = clock
        self.name = name
        self.csv_enabled = bool(csv_enabled)
        self.verbose = bool(verbose)
        self.cash = cfg.start_capital
        self.pos: dict[str, Any] | None = None
        self.trades: list[dict[str, Any]] = []
        self.watch: dict[str, dict[str, Any]] = {}
        self.candidates: dict[tuple, dict[str, Any]] = {}
        self.csv_path = csv_path or f"./trades_{name.lower()}_dash.csv"
        self.csv_ready = os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        self.day_start_cap: float | None = None
        self.current_day = None
        self.micro_exit_cooldown: dict[str, float] = {}
        self.micro_candidate_cooldown: dict[str, float] = {}
        self.hooks = hooks or TraderHooks()

    def now(self) -> float:
        return self.clock.now()

    # ── wallet / fills ──────────────────────────────────
    def _check_daily_loss(self, mark: float | None = None) -> bool:
        today = datetime.fromtimestamp(self.now()).date()
        current = self.capital_now(mark)
        
        if self.current_day != today or self.day_start_cap is None:
            self.day_start_cap = current
            self.current_day = today
            return False

        if self.day_start_cap > 0 and current / self.day_start_cap < (1 - self.cfg.daily_loss_limit):
            if self.verbose:
                print(f"🚨 DAILY LOSS LIMIT ({self.cfg.daily_loss_limit*100:.1f}%) HIT — KILL SWITCH")
        
            if self.pos:
                exit_px = mark if (mark is not None and mark > 0) else self.market.mark_for_exit(self.pos["pair"])
                self._exit(exit_px or self.pos["entry_signal"], "daily_loss_kill")
        
            return True
        
        return False

    def capital_now(self, mark: float | None = None) -> float:
        if not self.pos:
            return self.cash
        
        p = mark if mark and mark > 0 else self.market.mark_for_exit(self.pos["pair"])
        
        if p <= 0:
            p = self.pos["entry_signal"]
        
        return self.cash + self.pos["qty"] * p * (1 - self.cfg.slippage) * (1 - self.cfg.fee_pct)

    def net_ret_if_exit(self, price: float) -> float:
        if not self.pos or price <= 0:
            return 0.0

        gross_proceeds = self.market.estimate_taker_sell_fill(self.pos["pair"], self.pos["qty"])

        if gross_proceeds <= 0:
            gross_proceeds = self.pos["qty"] * price * (1 - self.cfg.slippage)

        proceeds = gross_proceeds * (1 - self.cfg.fee_pct)
        return proceeds / self.pos["entry_cost"] - 1.0

    def _current_heat(self) -> float:
        if not self.pos:
            return 0.0

        cap = self.capital_now()
        mark = self.market.mark_for_exit(self.pos["pair"]) or self.pos["entry_signal"]
        exposure = self.pos["qty"] * mark
        
        return exposure / cap if cap > 0 else 0.0

    def get_position_size(self, symbol: str, regime: str, score: float) -> float:
        if self.cash <= 0:
            return 0.0
        
        equity = self.capital_now()
        min_ticket = self.cfg.min_ticket_usd
        max_by_cash = self.cash
        max_by_heat = max(0.0, self.cfg.max_portfolio_heat * equity)
        
        if min(max_by_cash, max_by_heat) < min_ticket:
            return 0.0

        base = equity * 0.28
        vol_factor = 1.0
        hist = self.market.latest_closed(symbol)[-20:]
        lp = self.market.last_price.get(symbol, 0.0)
        
        if hist and lp > 0:
            atr = np.mean([abs(c[2] - c[3]) for c in hist])
            atr_pct = atr / lp
            vol_factor = max(0.4, min(1.55, self.cfg.vol_target_atr / (atr_pct + 1e-9)))

        liq_factor = 1.0
        
        if self.market.book_is_fresh(symbol):
            top_not = sum(p * q for p, q in self.market.top_bids(symbol, 5))
            liq_factor = min(1.0, top_not / self.cfg.liq_buffer)

        score_factor = clamp(score / 9.0, 0.65, 1.35)
        raw = base * vol_factor * liq_factor * score_factor
        size = max(min_ticket, raw)
        size = min(size, max_by_cash, max_by_heat)
        
        return size if size >= min_ticket else 0.0

    def _csv_schema_ok(self, rec: dict[str, Any]) -> bool:
        if not self.csv_ready or not os.path.exists(self.csv_path):
            return True
        
        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            return header == list(rec.keys())
        except Exception:
            return False

    def _csv_append(self, rec: dict[str, Any]) -> None:
        if not self._csv_schema_ok(rec):
            rotated = self.csv_path + ".old_schema"
        
            try:
                os.replace(self.csv_path, rotated)
                if self.verbose:
                    print(f"[CSV] rotated old-schema trade log to {rotated}")
            except OSError:
                pass
        
            self.csv_ready = False
        
        mode = "a" if self.csv_ready else "w"
        
        with open(self.csv_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=rec.keys())
        
            if not self.csv_ready:
                w.writeheader()
                self.csv_ready = True
        
            w.writerow(rec)
            f.flush()

    def _enter(self, pair: str, signal_price: float, regime: str, score: float, note: str = "") -> None:
        if signal_price <= 0 or self.cash <= 0:
            return
        
        if self._check_daily_loss():
            return

        qty_cap = self.get_position_size(pair, regime, score)
        
        if qty_cap <= 0:
            return
        
        fill, gross_qty = self.market.estimate_taker_buy_fill(pair, qty_cap)
        
        if fill <= 0 or gross_qty <= 0:
            return
        
        qty = gross_qty * (1 - self.cfg.fee_pct)

        self.pos = {
            "pair": pair,
            "entry": fill,
            "entry_signal": signal_price,
            "qty": qty,
            "entry_cost": qty_cap,
            "entry_time": self.now(),
            "peak_mark": signal_price,
            "trail_active": False,
            "regime": regime,
            "score": score,
            "note": note,
        }
        
        self.cash -= qty_cap
        self.market.mark_entry(self.now(), pair, fill, regime)

        if self.verbose:
            t = datetime.now().strftime("%H:%M:%S")
        
            print(
                f"[{t}] [{self.name:12s}] BUY  {pair} @ {fill:.6g} [{regime}] "
                f"score={score:.2f} size=${qty_cap:.2f} ({note})"
            )
        
        self.hooks.fill()

    def _exit(self, signal_price: float, reason: str) -> None:
        if not self.pos or signal_price <= 0:
            return
        
        pair = self.pos["pair"]
        gross_proceeds = self.market.estimate_taker_sell_fill(pair, self.pos["qty"])
        
        if gross_proceeds <= 0:
            gross_proceeds = self.pos["qty"] * signal_price * (1 - self.cfg.slippage)
        
        fill = gross_proceeds / self.pos["qty"] if self.pos["qty"] > 0 else signal_price
        proceeds = gross_proceeds * (1 - self.cfg.fee_pct)
        pnl = proceeds - self.pos["entry_cost"]
        hold_sec = self.now() - self.pos["entry_time"]
        self.cash += proceeds
        trade_ret = pnl / max(self.pos["entry_cost"], 1e-9)
        hold_min = max(hold_sec / 60.0, 1.0 / 60.0)
        return_velocity_per_min = trade_ret / hold_min

        rec = {
            "pair": pair,
            "regime": self.pos["regime"],
            "score": self.pos["score"],
            "entry_time": iso_from_ts(self.pos["entry_time"]),
            "exit_time": iso_from_ts(self.now()),
            "hold_sec": hold_sec,
            "entry": self.pos["entry"],
            "exit": fill,
            "qty": self.pos["qty"],
            "entry_cost": self.pos["entry_cost"],
            "reason": reason,
            "pnl_usd": pnl,
            "return_pct": trade_ret * 100.0,
            "return_velocity_pct_per_min": return_velocity_per_min * 100.0,
            "capital_after": self.cash,
            "note": self.pos["note"],
        }
        
        self.trades.append(rec)
        
        if self.csv_enabled:
            try:
                self._csv_append(rec)
            except Exception as e:
                if self.verbose:
                    print(f"  [CSV] {e}")

        if self.pos["regime"] == "book_ignition":
            self.micro_exit_cooldown[pair] = self.now()

        self.market.mark_exit(self.now(), pair, fill, reason, pnl)
        
        if self.verbose:
            t = datetime.now().strftime("%H:%M:%S")
        
            print(
                f"[{t}] [{self.name:12s}] SELL {pair} @ {fill:.6g} "
                f"reason={reason} P&L=${pnl:+.4f} cap=${self.cash:.4f}"
            )
        
        self.pos = None
        self.candidates.clear()
        self.hooks.fill()

    # ── candidate queue ─────────────────────────────────
    def push(self, key: tuple, candidate: dict[str, Any]) -> None:
        candidate["raw_score"] = candidate.get("raw_score", candidate["score"])
        existing = self.candidates.get(key)
        
        if not existing or candidate["score"] > existing["score"]:
            self.candidates[key] = candidate

    def execute_due(self) -> None:
        if self.pos:
            return
        
        tnow = self.now()
        
        for c in self.candidates.values():
            age = tnow - c["at"]
            c["score"] = c.get("raw_score", c["score"]) * max(0.65, 1.0 - age / 45.0)

        stale = [k for k, c in self.candidates.items() if tnow - c["at"] > self.cfg.max_candidate_age]
        
        if stale:
            self.hooks.candidate_expired(len(stale))
        
            for k in stale:
                self.candidates.pop(k, None)

        due = [c for c in self.candidates.values() if tnow - c["at"] >= self.cfg.candidate_debounce_sec]
        
        if not due:
            return
        
        due.sort(key=lambda x: x["score"], reverse=True)
        self.candidates.clear()

        for cand in due:
            symbol = cand["symbol"]
            current = self.market.mark_for_entry(symbol)
        
            if current <= 0:
                self.hooks.candidate_rejected(1)
                continue
        
            if current > cand["price"] * (1.0 + cand.get("max_chase", 0.006)):
                self.hooks.candidate_rejected(1)
                continue
        
            if current < cand["price"] * (1.0 - cand.get("fail_drop", 0.004)):
                self.hooks.candidate_rejected(1)
                continue
        
            if cand["regime"] == "book_ignition":
                sp = self.market.spread_bps(symbol)
                imb = self.market.book_imbalance(symbol, 5)
        
                if sp is None or imb is None or sp > cand.get("max_spread_bps", 12.0) or imb < 1.05:
                    self.hooks.candidate_rejected(1)
                    continue
            elif self.cfg.macro_require_microstructure and not self.market.live_microstructure_ok(symbol):
                self.hooks.candidate_rejected(1)
                continue
            
            self._enter(symbol, current, cand["regime"], cand["score"], cand["note"])
            
            if cand["regime"] == "macro_reclaim_v2":
                self.watch.pop(symbol, None)
            
            return

    # ── risk exits ──────────────────────────────────────
    def on_tick(self, symbol: str, price: float | None = None) -> None:
        if not self.pos or self.pos["pair"] != symbol:
            return
        
        mark = self.market.mark_for_exit(symbol) or price or 0.0
        
        if mark <= 0:
            return
        
        if self._check_daily_loss(mark):
            return

        r = self.cfg.risk(self.pos["regime"])
        net_ret = self.net_ret_if_exit(mark)

        if mark > self.pos["peak_mark"]:
            self.pos["peak_mark"] = mark

        if net_ret <= -r["hard_stop"]:
            self._exit(mark, "hard_stop_net")
            return

        if r.get("take_profit") is not None and net_ret >= r["take_profit"]:
            self._exit(mark, "take_profit")
            return

        age = self.now() - self.pos["entry_time"]
        age_min = max(age / 60.0, 1.0 / 60.0)
        return_velocity_per_min = net_ret / age_min
        
        if self.pos["regime"] in ("macro_thrust", "macro_reclaim_v2"):
            if age >= self.cfg.macro_early_fail_sec and net_ret <= self.cfg.macro_early_fail_ret:
                self._exit(mark, "macro_early_fail")
                return
        
            if age >= self.cfg.macro_no_followthrough_sec and net_ret < self.cfg.macro_no_followthrough_ret:
                self._exit(mark, "macro_no_followthrough")
                return

        efficiency_check = min(float(r["max_hold"]) * 0.55, self.cfg.time_efficiency_check_sec)
        if (
            age >= efficiency_check
            and net_ret < self.cfg.time_efficiency_min_ret
            and return_velocity_per_min < self.cfg.min_return_velocity_per_min
        ):
            self._exit(mark, "return_velocity_stall")
            return

        if not self.pos["trail_active"] and net_ret >= r["trail_activate"]:
            self.pos["trail_active"] = True
        
            if self.verbose:
                t = datetime.now().strftime("%H:%M:%S")
                print(f"[{t}] [{self.name:12s}] Trail ON {symbol} net {net_ret*100:+.2f}%")

        trail_pct = r["trail_pct"]
        tightens_at = min(float(r["max_hold"]) * 0.55, self.cfg.horizon_trail_tighten_sec)
        if age >= tightens_at and net_ret > self.cfg.time_efficiency_min_ret:
            trail_pct = min(trail_pct, self.cfg.horizon_tight_trail_pct)

        if self.pos["trail_active"] and mark <= self.pos["peak_mark"] * (1 - trail_pct):
            reason = "horizon_tight_trail" if trail_pct < r["trail_pct"] else "trail"
            self._exit(mark, reason)
            return

        if age >= r["max_hold"]:
            self._exit(mark, "timeout")
            return

        if self.pos and self.pos["regime"] == "book_ignition" and self.market.book_is_fresh(symbol):
            imb = self.market.book_imbalance(symbol, 5)
            sp = self.market.spread_bps(symbol)
        
            if age > 25 and imb is not None and sp is not None and imb < 0.70 and net_ret < 0.010:
                self._exit(mark, "book_flip")
                return

        if self.pos and age >= r["stall_check"] and net_ret < r["stall_ret"]:
            self._exit(mark, "stall_net")

    def on_book(self, symbol: str) -> None:
        if self.pos and self.pos["pair"] == symbol:
            self.on_tick(symbol, self.market.mark_for_exit(symbol))

    def cleanup(self) -> None:
        tnow = self.now()
        stale = [s for s, w in self.watch.items() if tnow - w["at"] > self.cfg.watch_max_age]
        
        for s in stale:
            del self.watch[s]
        
        for sym in list(self.market.tape.keys()):
            self.market.prune_tape(sym, tnow)
        
        self.market.recalc_running_sums()
        stale_c = [k for k, c in self.candidates.items() if tnow - c["at"] > self.cfg.max_candidate_age]
        
        if stale_c:
            self.hooks.candidate_expired(len(stale_c))
        
            for k in stale_c:
                self.candidates.pop(k, None)
        
        self._check_daily_loss()

    # ── signals ─────────────────────────────────────────
    def detect_pump(self, symbol: str) -> dict[str, Any] | None:
        hist = self.market.latest_closed(symbol)
        
        if len(hist) < self.cfg.baseline_mins + 1:
            return None
        
        sample = hist[-(self.cfg.baseline_mins + 1):]
        *baseline, latest = sample
        base_vol = median([c[5] for c in baseline], 0.0)
        
        if base_vol <= 0:
            return None
        
        base_lows = [c[3] for c in baseline]
        base_highs = [c[2] for c in baseline]
        min_low = min(base_lows) if base_lows else 0.0
        
        if min_low <= 0:
            return None
        
        candle_range = latest[2] - latest[3]
        spike_pct = (candle_range / latest[3] * 100.0) if latest[3] else 0.0
        
        if spike_pct < self.cfg.min_spike_pct:
            return None
        
        base_range = (max(base_highs) - min_low) / min_low * 100.0
        
        if base_range > spike_pct * 0.55:
            return None
        
        burst_x = latest[5] / base_vol
        
        if burst_x < self.cfg.vol_burst_x:
            return None
        
        if candle_range > 0 and close_position(latest) < self.cfg.pump_candle_upper_pct:
            return None
        
        tnow = self.now()
        last = self.market.last_pump_alert.get(symbol)
        
        if last is not None and tnow - last < self.cfg.cooldown_sec:
            return None
        
        self.market.last_pump_alert[symbol] = tnow
        return {"spike_pct": spike_pct, "burst_x": burst_x, "high": latest[2], "close": latest[4], "volume": latest[5]}

    def on_pump(self, symbol: str, sig: dict[str, Any]) -> None:
        if symbol == BTC_SYMBOL or self.pos:
            return
        
        if symbol not in self.watch:
            self.watch[symbol] = {"anchor": sig["high"], "at": self.now(), "spike_pct": sig["spike_pct"], "burst_x": sig["burst_x"]}
        
            if self.verbose:
                t = datetime.now().strftime("%H:%M:%S")
                print(f"[{t}] [WATCH        ] {symbol} pump +{sig['spike_pct']:.1f}% vol x{sig['burst_x']:.1f}")

    def maybe_reclaim_candidate(self, symbol: str, c: dict[str, Any], minute_key: Any) -> None:
        if self.pos or symbol == BTC_SYMBOL or symbol not in self.watch:
            return
        
        if not self.market.btc_context_ok(strict=False):
            return
        
        if self.cfg.macro_require_microstructure and not self.market.live_microstructure_ok(symbol):
            return

        tnow = self.now()
        w = self.watch[symbol]
        
        if tnow - w["at"] <= self.cfg.peak_update_window and c["high"] > w["anchor"]:
            w["anchor"] = c["high"]
        
        anchor = w["anchor"]
        
        if anchor <= 0:
            return
        
        drop = 1.0 - c["close"] / anchor
        
        if tnow - w["at"] > self.cfg.watch_max_age or drop >= self.cfg.invalidate_drop:
            self.watch.pop(symbol, None)
            return
        
        if drop < self.cfg.pullback_min or drop > self.cfg.pullback_max:
            return

        hist = self.market.latest_closed(symbol)
        
        if len(hist) < self.cfg.reclaim_lookback + 1:
            return
        
        recent = hist[-(self.cfg.reclaim_lookback + 1):-1]
        r_closes = [x[4] for x in recent]
        r_vols = [x[5] for x in recent]
        prev_close = r_closes[-1] if r_closes else 0.0
        mean_close = sum(r_closes) / len(r_closes) if r_closes else 0.0
        mean_vol = median(r_vols, 0.0)
        vol_r = c["volume"] / mean_vol if mean_vol > 0 else 0.0
        cp = candle_close_pos_dict(c)
        
        if (
            c["close"] > c["open"]
            and c["close"] > prev_close
            and c["close"] > mean_close
            and cp >= 0.65
            and vol_r >= self.cfg.reclaim_vol_floor
            and self.market.book_quality_ok(symbol, max_spread_bps=35.0, min_imbalance=0.70, require_book=self.cfg.macro_require_microstructure)
        ):
            sweet = 0.085
            pb_q = clamp(1.0 - abs(drop - sweet) / 0.10)
        
            score = (
                min(w["spike_pct"] / 28.0, 1.0) * 2.0
                + min(w["burst_x"] / 18.0, 1.0) * 2.0
                + pb_q * 2.5
                + min(vol_r / 3.0, 1.0) * 2.5
                + cp * 1.0
            )
        
            self.push(("macro_reclaim_v2", minute_key, symbol), {
                "symbol": symbol,
                "price": c["close"],
                "regime": "macro_reclaim_v2",
                "score": score,
                "note": f"reclaim pb {drop*100:.1f}% vol {vol_r:.2f}x cp {cp:.2f}",
                "at": self.now(),
                "max_chase": 0.012,
                "fail_drop": 0.006,
            })

    def maybe_thrust_candidate(self, symbol: str, c: dict[str, Any], minute_key: Any) -> None:
        if self.pos or not self.market.btc_context_ok(strict=False):
            return

        hist = self.market.latest_closed(symbol)
        
        if len(hist) < self.cfg.thrust_lookback + 1:
            return
        
        if self.cfg.macro_require_microstructure and not self.market.live_microstructure_ok(symbol):
            return

        close = c["close"]
        
        if close <= 0:
            return
        
        r3 = close / hist[-4][4] - 1.0 if len(hist) >= 4 and hist[-4][4] > 0 else 0.0
        r5 = close / hist[-6][4] - 1.0 if len(hist) >= 6 and hist[-6][4] > 0 else 0.0
        r15 = close / hist[-16][4] - 1.0 if len(hist) >= 16 and hist[-16][4] > 0 else 0.0
        
        if r3 < self.cfg.thrust_min_ret_3m or r5 < self.cfg.thrust_min_ret_5m or r5 > self.cfg.thrust_max_ret_5m or r15 < 0:
            return
        
        prev = hist[-(self.cfg.thrust_lookback + 1):-1]
        med_vol = median([x[5] for x in prev], 0.0)
        
        if med_vol <= 0:
            return
        
        vol_x = c["volume"] / med_vol
        
        if vol_x > self.cfg.macro_max_vol_x:
            return
        
        cp = candle_close_pos_dict(c)
        
        if vol_x < self.cfg.thrust_min_vol_x or cp < self.cfg.thrust_min_close_pos or c["close"] <= c["open"]:
            return
        
        candle_ret = c["close"] / c["open"] - 1.0 if c["open"] > 0 else 0.0
        
        if candle_ret < 0.004:
            return
        
        if not self.market.book_quality_ok(symbol, max_spread_bps=self.cfg.macro_max_spread_bps, min_imbalance=self.cfg.macro_min_book_imbalance, require_book=self.cfg.macro_require_microstructure):
            return
        
        if abs(r5) < self.market.estimated_cost_floor(symbol) * 1.5:
            return

        sp = self.market.spread_bps(symbol)
        sp_penalty = 0.0 if sp is None else clamp(sp / max(self.cfg.macro_max_spread_bps, 1.0))
        score = (
            3.0 * clamp(r5 / 0.07)
            + 2.0 * clamp(r3 / 0.035)
            + 2.5 * clamp(vol_x / 8.0)
            + 1.5 * cp
            + 1.0 * clamp(candle_ret / 0.025)
            - 0.8 * sp_penalty
        )
        self.push(("macro_thrust", minute_key, symbol), {
            "symbol": symbol,
            "price": c["close"],
            "regime": "macro_thrust",
            "score": score,
            "note": f"thrust r5 {r5*100:.2f}% r3 {r3*100:.2f}% vol {vol_x:.1f}x cp {cp:.2f}",
            "at": self.now(),
            "max_chase": 0.010,
            "fail_drop": 0.006,
        })

    def maybe_micro_candidate(self, symbol: str) -> None:
        if symbol in self.market.trade_symbols:
            snap = self.market.snapshot_micro(symbol)
        
            if snap:
                self.market.remember_micro_snapshot(symbol, snap)
        
        if self.pos or symbol not in self.market.trade_symbols or not self.market.book_is_fresh(symbol):
            return

        tnow = self.now()
        
        if symbol in self.micro_exit_cooldown and tnow - self.micro_exit_cooldown[symbol] < self.cfg.micro_cooldown_sec * 0.7:
            return
        
        if symbol in self.micro_candidate_cooldown and tnow - self.micro_candidate_cooldown[symbol] < self.cfg.micro_candidate_cooldown_sec:
            return

        snap = self.market.snapshot_micro(symbol)
        
        if not snap or snap["trades"] < self.cfg.micro_min_trades:
            return
        
        if snap["buy_not"] < self.cfg.micro_min_buy_notional:
            return

        self.market.prune_tape(symbol, tnow)
        
        with self.market.lock:
            dq = list(self.market.tape[symbol])
        
        if len(dq) < 8:
            return
        
        prices = [x[2] for x in dq[-10:]]
        
        if len(prices) < 8:
            return
        
        p0, p1, p2 = prices[-8], prices[-4], prices[-1]
        
        if p0 <= 0 or p1 <= 0:
            return
        
        r_prev = p1 / p0 - 1.0
        r_now = p2 / p1 - 1.0
        
        if r_now <= 0:
            return
        
        accel = r_now / max(abs(r_prev), 0.0005)

        move_pct = snap["move_pct"]
        flow = snap["buy_not"] + snap["sell_not"]
        buy_share = snap["buy_not"] / max(flow, 1.0)
        delta_div = buy_share / max(snap.get("baseline_buy_share", 0.5), 0.05)
        ms = snap.get("max_spread", 18.0)

        if move_pct < self.market.estimated_cost_floor(symbol) * 0.75:
            return

        if (
            snap["burst_ratio"] >= self.cfg.micro_burst_multiple
            and snap["imbalance"] >= self.cfg.micro_min_imbalance
            and move_pct >= self.cfg.micro_min_move_pct
            and accel >= self.cfg.micro_accel_threshold
            and delta_div >= self.cfg.micro_delta_divergence
            and snap["book_imb"] is not None and snap["book_imb"] >= 1.08
            and snap["spread_bps"] is not None and snap["spread_bps"] <= ms
        ):
            score = (
                3.0 * clamp(snap["burst_ratio"] / 9.0)
                + 2.5 * clamp(snap["imbalance"] / 5.0)
                + 2.4 * clamp(move_pct / 0.015)
                + 1.8 * clamp(accel / 2.5)
                + 1.4 * clamp((snap["book_imb"] - 1.0) / 0.8)
                + 0.7 * clamp(delta_div / 1.8)
                - 0.7 * clamp(snap["spread_bps"] / ms)
            )

            self.push(("book_ignition", int(tnow / self.cfg.candidate_debounce_sec), symbol), {
                "symbol": symbol,
                "price": dq[-1][2],
                "regime": "book_ignition",
                "score": score,
                "note": f"micro {accel:.2f}x Δ{delta_div:.2f}",
                "at": tnow,
                "max_chase": 0.005,
                "fail_drop": 0.004,
                "max_spread_bps": ms,
            })
            
            self.micro_candidate_cooldown[symbol] = tnow

    def on_closed_candle(self, symbol: str, c: dict[str, Any], minute_key: Any) -> None:
        tnow = self.now()
        
        if symbol in self.watch:
            w = self.watch[symbol]
        
            if tnow - w["at"] <= self.cfg.peak_update_window and c["high"] > w["anchor"]:
                w["anchor"] = c["high"]
        
            drop = 1.0 - c["close"] / w["anchor"] if w["anchor"] else 0.0
        
            if tnow - w["at"] > self.cfg.watch_max_age or drop >= self.cfg.invalidate_drop:
                self.watch.pop(symbol, None)

        if self.pos and self.pos["pair"] == symbol:
            self.on_tick(symbol, c["close"])
        
            if not self.pos:
                return
        
            age = tnow - self.pos["entry_time"]
            net_ret = self.net_ret_if_exit(self.market.mark_for_exit(symbol) or c["close"])
            cp = candle_close_pos_dict(c)
        
            if symbol != BTC_SYMBOL and self.market.btc_flush_active() and net_ret < 0.020:
                self._exit(self.market.mark_for_exit(symbol) or c["close"], "btc_flush")
                return
        
            if self.pos and self.pos["regime"] == "book_ignition":
                if age > 90 and c["close"] < c["open"] and cp < 0.35 and net_ret < 0.015:
                    self._exit(self.market.mark_for_exit(symbol) or c["close"], "micro_candle_fail")
                    return
        
        if self.pos:
            return

        sig = self.detect_pump(symbol)
        
        if sig:
            self.on_pump(symbol, sig)
        
        self.maybe_reclaim_candidate(symbol, c, minute_key)
        self.maybe_thrust_candidate(symbol, c, minute_key)

    def summary_metrics(self, n_msgs: int = 0) -> dict[str, Any]:
        final = self.cash if not self.pos else self.capital_now()
        pnls = [float(t["pnl_usd"]) for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        total_pnl = sum(pnls)

        peak = self.cfg.start_capital
        max_dd = 0.0
        current_dd = 0.0
        with self.market.lock:
            caps = list(self.market.capital_history)
        for _, cap in caps:
            peak = max(peak, cap)
            dd = (peak - cap) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            current_dd = dd
        if final > peak:
            peak = final
            current_dd = 0.0
        elif peak > 0:
            current_dd = (peak - final) / peak
            max_dd = max(max_dd, current_dd)

        by_regime = defaultdict(lambda: {
            "n": 0,
            "wins": 0,
            "pnl": 0.0,
            "hold_sec": 0.0,
            "return_pct": 0.0,
            "return_velocity_pct_per_min": 0.0,
        })
        for t in self.trades:
            b = by_regime[t["regime"]]
            b["n"] += 1
            b["pnl"] += float(t.get("pnl_usd", 0.0) or 0.0)
            b["hold_sec"] += float(t.get("hold_sec", 0.0) or 0.0)
            b["return_pct"] += float(t.get("return_pct", 0.0) or 0.0)
            b["return_velocity_pct_per_min"] += float(t.get("return_velocity_pct_per_min", 0.0) or 0.0)
            if float(t.get("pnl_usd", 0.0) or 0.0) > 0:
                b["wins"] += 1
        by_regime_out = {}
        for k, v in by_regime.items():
            n = max(int(v["n"]), 1)
            row = dict(v)
            row["avg_hold_sec"] = row["hold_sec"] / n
            row["avg_return_pct"] = row["return_pct"] / n
            row["avg_return_velocity_pct_per_min"] = row["return_velocity_pct_per_min"] / n
            by_regime_out[k] = row

        duration_hours = 1 / 60.0
        if len(caps) >= 2:
            duration_hours = max((caps[-1][0] - caps[0][0]) / 3600.0, 1 / 60.0)

        closed_holds = [float(t.get("hold_sec", 0.0) or 0.0) for t in self.trades]
        open_hold_sec = (self.now() - self.pos["entry_time"]) if self.pos else 0.0
        exposure_hours = (sum(closed_holds) + max(open_hold_sec, 0.0)) / 3600.0
        exposure_ratio = min(exposure_hours / duration_hours, 1.0) if duration_hours > 0 else 0.0
        avg_hold_sec = sum(closed_holds) / len(closed_holds) if closed_holds else 0.0
        median_hold_sec = median(closed_holds, 0.0)
        win_holds = [float(t.get("hold_sec", 0.0) or 0.0) for t in self.trades if float(t.get("pnl_usd", 0.0) or 0.0) > 0]
        loss_holds = [float(t.get("hold_sec", 0.0) or 0.0) for t in self.trades if float(t.get("pnl_usd", 0.0) or 0.0) <= 0]
        trade_returns = [float(t.get("return_pct", 0.0) or 0.0) / 100.0 for t in self.trades]
        trade_velocities = [float(t.get("return_velocity_pct_per_min", 0.0) or 0.0) / 100.0 for t in self.trades]

        net_return = (final / self.cfg.start_capital - 1.0) if self.cfg.start_capital else 0.0
        return_per_hour = net_return / duration_hours if duration_hours > 0 else 0.0
        profit_per_hour = total_pnl / duration_hours if duration_hours > 0 else 0.0
        return_per_exposure_hour = net_return / exposure_hours if exposure_hours > 0 else 0.0
        pnl_per_exposure_hour = total_pnl / exposure_hours if exposure_hours > 0 else 0.0
        avg_trade_velocity = sum(trade_velocities) / len(trade_velocities) if trade_velocities else 0.0
        median_trade_velocity = median(trade_velocities, 0.0)
        avg_trade_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
        median_trade_return = median(trade_returns, 0.0)
        avg_hold_min = avg_hold_sec / 60.0
        median_hold_min = median_hold_sec / 60.0
        horizon_score = (
            100.0 * net_return
            + 35.0 * return_per_hour
            + 20.0 * return_per_exposure_hour
            + 20.0 * avg_trade_velocity
            - 120.0 * max_dd
            - 0.60 * avg_hold_min
            - 0.40 * median_hold_min
        )

        return {
            "messages": n_msgs,
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total) if total else 0.0,
            "total_pnl": total_pnl,
            "best_trade": max(pnls) if pnls else 0.0,
            "worst_trade": min(pnls) if pnls else 0.0,
            "best_trade_return_pct": max(trade_returns) * 100.0 if trade_returns else 0.0,
            "worst_trade_return_pct": min(trade_returns) * 100.0 if trade_returns else 0.0,
            "final_capital": final,
            "net_return": net_return,
            "max_drawdown": max_dd,
            "current_drawdown": current_dd,
            "duration_hours": duration_hours,
            "profit_per_hour": profit_per_hour,
            "return_per_hour": return_per_hour,
            "return_per_exposure_hour": return_per_exposure_hour,
            "pnl_per_exposure_hour": pnl_per_exposure_hour,
            "avg_hold_sec": avg_hold_sec,
            "median_hold_sec": median_hold_sec,
            "avg_win_hold_sec": sum(win_holds) / len(win_holds) if win_holds else 0.0,
            "avg_loss_hold_sec": sum(loss_holds) / len(loss_holds) if loss_holds else 0.0,
            "open_hold_sec": max(open_hold_sec, 0.0),
            "exposure_hours": exposure_hours,
            "exposure_ratio": exposure_ratio,
            "avg_trade_return_pct": avg_trade_return * 100.0,
            "median_trade_return_pct": median_trade_return * 100.0,
            "avg_return_velocity_pct_per_min": avg_trade_velocity * 100.0,
            "median_return_velocity_pct_per_min": median_trade_velocity * 100.0,
            "horizon_score": horizon_score,
            "by_regime": by_regime_out,
        }

    def performance_snapshot(self, n_msgs: int = 0) -> dict[str, Any]:
        m = self.summary_metrics(n_msgs)
        return {
            "net_return": m["net_return"],
            "return_per_hour": m["return_per_hour"],
            "return_per_exposure_hour": m["return_per_exposure_hour"],
            "current_drawdown": m["current_drawdown"],
            "horizon_score": m["horizon_score"],
            "avg_hold_sec": m["avg_hold_sec"],
            "median_hold_sec": m["median_hold_sec"],
            "exposure_ratio": m["exposure_ratio"],
            "avg_return_velocity_pct_per_min": m["avg_return_velocity_pct_per_min"],
            "trades": m["trades"],
        }

