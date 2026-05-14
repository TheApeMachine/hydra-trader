from __future__ import annotations

import copy
import dataclasses
import math
import threading
from collections import defaultdict, deque
from typing import Any

import numpy as np

from .config import Config
from .constants import (
    BOOK_DEPTH,
    BTC_SYMBOL,
    DASH_CAPITAL_LEN,
    DASH_FLOW_LEN,
    DASH_PERF_LEN,
    DASH_PRICE_LEN,
    DASH_TAPE_LEN,
    STATE_LEN,
)
from . import flow_metrics
from .utils import clamp, close_position, median, sf


@dataclasses.dataclass(frozen=True)
class ClosedCandleEvent:
    symbol: str
    candle: dict[str, float | str]
    minute_key: Any


@dataclasses.dataclass(frozen=True)
class MarketView:
    """Immutable-enough copy used by the dashboard.

    The snapshot intentionally carries tuples/lists copied under MarketStore's
    lock so the UI never iterates live deques or mutable book dictionaries.
    """

    ts: float
    focus: str
    price_series: tuple[tuple[float, float], ...]
    tape_series: tuple[tuple[float, float, float], ...]
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    entry_marks: tuple[tuple, ...]
    exit_marks: tuple[tuple, ...]
    capital_history: tuple[tuple[float, float], ...]
    micro: dict[str, Any]
    thrust: dict[str, Any]
    last_price: float
    spread_bps: float | None
    book_imbalance: float | None
    # (ts, churn, visc_raw, turb_var, accel, signed_$_per_s) per sample
    flow_series: tuple[tuple[float, float, float, float, float, float], ...]
    # (ts, net_ret, ret_per_hour, ret_per_exposure_hour, drawdown, horizon_score, avg_hold_sec, exposure_ratio, avg_return_velocity_pct_per_min, trades)
    performance_history: tuple[tuple[float, float, float, float, float, float, float, float, float, float], ...]
    # symbol -> ((ts, price), ...) for the multi-symbol chart
    multi_prices: dict[str, tuple[tuple[float, float], ...]]
    # (ts, excitation, buy_sell_ratio, slope) for focus
    hawkes_history: tuple[tuple[float, float, float, float], ...]
    # (ts_made, predicted_return_pct, horizon_sec) for focus
    hawkes_predictions: tuple[tuple[float, float, float], ...]
    # symbol -> ((ts, excitation, ratio, slope), ...) — per-symbol Hawkes history
    multi_hawkes_history: dict[str, tuple[tuple[float, float, float, float], ...]]
    # symbol -> ((ts_made, predicted_return_pct, horizon_sec), ...)
    multi_hawkes_predictions: dict[str, tuple[tuple[float, float, float], ...]]
    # symbol -> {"n_matured": int, "hit_rate": float, "loss_ema": float, "k": float,
    #            "slope_weight": float, "bias": float, "status": "cold|learning|calibrated|degraded"}
    hawkes_learning: dict[str, dict[str, Any]]
    # symbol -> {"pressure": -1|0|1, "readiness": -1|0|1} sticky inertia states
    inertia_state: dict[str, dict[str, int]]
    # global backdrop inertia: -1 (weak/flush), 0 (neutral), +1 (supportive)
    btc_backdrop_state: int


class MarketStore:
    """Owns all mutable market state.

    Strategy and UI code interact with this object through methods. This is the
    key v5 boundary: globals are gone, and every live/replay/Optuna engine gets
    an independent MarketStore instance.
    """

    def __init__(
        self,
        cfg: Config,
        clock,
        book_symbols: list[str] | None = None,
        trade_symbols: list[str] | None = None,
        ohlc_only: bool = False,
    ):
        self.cfg = cfg
        self.clock = clock
        self.lock = threading.RLock()
        self.ohlc_only = bool(ohlc_only)
        self.book_symbols = set(book_symbols or [])
        self.trade_symbols = set(trade_symbols or [])

        self.state = defaultdict(lambda: deque(maxlen=STATE_LEN))
        self.closed_state = defaultdict(lambda: deque(maxlen=STATE_LEN))
        self.symbol_live = defaultdict(bool)
        self.last_price: dict[str, float] = {}
        self.btc_recent = deque(maxlen=30)

        self.tape = defaultdict(deque)
        self.tape_buy_sum = defaultdict(float)
        self.tape_sell_sum = defaultdict(float)
        self.base_tape = defaultdict(deque)
        self.base_buy_sum = defaultdict(float)
        self.base_total_sum = defaultdict(float)
        self.hawkes_state: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "buy": 0.0,
                "sell": 0.0,
                "last_ts": 0.0,
                "slope": 0.0,
                "last_pred_ts": 0.0,
            }
        )
        self.hawkes_base = defaultdict(deque)
        self.hawkes_base_buy_sum = defaultdict(float)
        self.hawkes_base_sell_sum = defaultdict(float)

        self.books: dict[str, dict[str, Any]] = {}
        self.book_ready = defaultdict(bool)

        self.last_pump_alert: dict[str, float] = {}

        self.capital_history = deque(maxlen=DASH_CAPITAL_LEN)
        self.performance_history = deque(maxlen=DASH_PERF_LEN)
        self.price_history = defaultdict(lambda: deque(maxlen=DASH_PRICE_LEN))
        self.tape_pressure_history = defaultdict(lambda: deque(maxlen=DASH_TAPE_LEN))
        self.entry_marks = deque(maxlen=200)
        self.exit_marks = deque(maxlen=200)
        self.decision_snapshots: dict[str, dict[str, Any]] = {}
        self.flow_history: dict[str, deque[tuple[float, float, float, float, float, float]]] = defaultdict(
            lambda: deque(maxlen=DASH_FLOW_LEN)
        )
        # (ts, excitation, buy_sell_ratio, slope) per symbol
        self.hawkes_history: dict[str, deque[tuple[float, float, float, float]]] = defaultdict(
            lambda: deque(maxlen=DASH_PRICE_LEN)
        )
        # (ts_made, predicted_return_pct, horizon_sec) per symbol — what Hawkes said would happen
        self.hawkes_predictions: dict[str, deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=DASH_PRICE_LEN)
        )
        # Predictions still waiting on their outcome:
        # (ts_made, predicted_pct, price_at_t, horizon, direction, magnitude, slope_norm)
        self.hawkes_predictions_pending: dict[str, deque[tuple[float, ...]]] = defaultdict(
            lambda: deque(maxlen=DASH_PRICE_LEN)
        )
        # Per-symbol online-learning state for the Hawkes prediction.
        # k, slope_weight, bias are the learnable parameters used in the formula:
        #   pred = k * direction * log1p(excess_excitation) * (1 + slope_weight * slope_norm) + bias
        self._hawkes_params_default = {
            "k": 0.0015,
            "slope_weight": 0.3,
            "bias": 0.0,
        }
        self.hawkes_learning: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "k": 0.0015,
                "slope_weight": 0.3,
                "bias": 0.0,
                "n_matured": 0,
                "hit_count": 0,
                "n_with_move": 0,
                "hit_rate": 0.0,
                "loss_ema": 0.0,
                "status": "cold",
                "last_pred_pct": 0.0,
                "last_actual_pct": 0.0,
            }
        )
        # Per-(symbol, signal_name) inertia trackers; lazy.
        from .utils import InertiaTracker
        self._inertia_cls = InertiaTracker
        self.inertia: dict[tuple[str, str], InertiaTracker] = {}
        # BTC backdrop inertia is global (one tracker).
        self.btc_backdrop_inertia = InertiaTracker(confirm_ticks=5)

    def now(self) -> float:
        return self.clock.now()

    def set_symbols(self, book_symbols: list[str] | None = None, trade_symbols: list[str] | None = None) -> None:
        with self.lock:
            if book_symbols is not None:
                self.book_symbols = set(book_symbols)
            if trade_symbols is not None:
                self.trade_symbols = set(trade_symbols)

    def reset(self) -> None:
        with self.lock:
            self.state.clear()
            self.closed_state.clear()
            self.symbol_live.clear()
            self.last_price.clear()
            self.btc_recent.clear()
            self.tape.clear()
            self.tape_buy_sum.clear()
            self.tape_sell_sum.clear()
            self.base_tape.clear()
            self.base_buy_sum.clear()
            self.base_total_sum.clear()
            self.hawkes_state.clear()
            self.hawkes_base.clear()
            self.hawkes_base_buy_sum.clear()
            self.hawkes_base_sell_sum.clear()
            self.books.clear()
            self.book_ready.clear()
            self.last_pump_alert.clear()
            self.capital_history.clear()
            self.performance_history.clear()
            self.price_history.clear()
            self.tape_pressure_history.clear()
            self.entry_marks.clear()
            self.exit_marks.clear()
            self.decision_snapshots.clear()
            self.flow_history.clear()
            self.hawkes_history.clear()
            self.hawkes_predictions.clear()
            self.hawkes_predictions_pending.clear()
            self.hawkes_learning.clear()
            for tr in self.inertia.values():
                tr.reset()
            self.inertia.clear()
            self.btc_backdrop_inertia.reset()

    # ── candle state ────────────────────────────────────
    def update_candle(self, c: dict[str, Any]) -> ClosedCandleEvent | None:
        sym = c.get("symbol")
        if not sym:
            return None
        ib = c.get("interval_begin")
        row = (
            ib,
            sf(c.get("open")),
            sf(c.get("high")),
            sf(c.get("low")),
            sf(c.get("close")),
            sf(c.get("volume")),
        )
        event = None
        tnow = self.now()
        with self.lock:
            if row[4] > 0:
                self.last_price[sym] = row[4]
                self.price_history[sym].append((tnow, row[4]))

            hist = self.state[sym]
            is_new = not hist or hist[-1][0] != ib

            if is_new and hist:
                cr = hist[-1]
                self.closed_state[sym].append(cr)
                if sym == BTC_SYMBOL:
                    self.btc_recent.append((cr[0], cr[4]))
                event = ClosedCandleEvent(
                    sym,
                    {"symbol": sym, "open": cr[1], "high": cr[2], "low": cr[3], "close": cr[4], "volume": cr[5]},
                    cr[0],
                )

                if sym in self.trade_symbols or sym == BTC_SYMBOL:
                    snap = self.snapshot_thrust_unlocked(sym)
                    if snap:
                        self.decision_snapshots.setdefault(sym, {}).update({f"thrust_{k}": v for k, v in snap.items()})

            if is_new:
                hist.append(row)
            else:
                hist[-1] = row
        return event

    def latest_closed(self, symbol: str) -> list[tuple]:
        with self.lock:
            return list(self.closed_state.get(symbol, []))

    # ── book helpers ────────────────────────────────────
    def _book_unlocked(self, sym: str) -> dict[str, Any]:
        return self.books.setdefault(sym, {"bids": {}, "asks": {}, "ts": 0.0})

    def truncate_book_unlocked(self, symbol: str, depth: int = BOOK_DEPTH) -> None:
        b = self.books.get(symbol)
        if not b:
            return
        b["bids"] = dict(sorted(b["bids"].items(), key=lambda x: -x[0])[:depth])
        b["asks"] = dict(sorted(b["asks"].items(), key=lambda x: x[0])[:depth])

    def apply_book(self, symbol: str, msg_type: str, bids: list[dict] | None, asks: list[dict] | None) -> None:
        if not symbol:
            return
        tnow = self.now()
        with self.lock:
            b = self._book_unlocked(symbol)
            if msg_type == "snapshot":
                b["bids"].clear()
                b["asks"].clear()

            for lvl in bids or []:
                price = sf(lvl.get("price"))
                qty = sf(lvl.get("qty"))
                if price <= 0:
                    continue
                if qty <= 0:
                    b["bids"].pop(price, None)
                else:
                    b["bids"][price] = qty

            for lvl in asks or []:
                price = sf(lvl.get("price"))
                qty = sf(lvl.get("qty"))
                if price <= 0:
                    continue
                if qty <= 0:
                    b["asks"].pop(price, None)
                else:
                    b["asks"][price] = qty

            self.truncate_book_unlocked(symbol, BOOK_DEPTH)
            b["ts"] = tnow
            if b["bids"] and b["asks"]:
                self.book_ready[symbol] = True

    def top_bids(self, symbol: str, levels: int = 10) -> list[tuple[float, float]]:
        with self.lock:
            return self.top_bids_unlocked(symbol, levels)

    def top_bids_unlocked(self, symbol: str, levels: int = 10) -> list[tuple[float, float]]:
        b = self.books.get(symbol)
        if not b:
            return []
        return sorted(b["bids"].items(), key=lambda x: -x[0])[:levels]

    def top_asks(self, symbol: str, levels: int = 10) -> list[tuple[float, float]]:
        with self.lock:
            return self.top_asks_unlocked(symbol, levels)

    def top_asks_unlocked(self, symbol: str, levels: int = 10) -> list[tuple[float, float]]:
        b = self.books.get(symbol)
        if not b:
            return []
        return sorted(b["asks"].items(), key=lambda x: x[0])[:levels]

    def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        with self.lock:
            return self.best_bid_ask_unlocked(symbol)

    def best_bid_ask_unlocked(self, symbol: str) -> tuple[float, float]:
        bids = self.top_bids_unlocked(symbol, 1)
        asks = self.top_asks_unlocked(symbol, 1)
        bid = bids[0][0] if bids else 0.0
        ask = asks[0][0] if asks else 0.0
        return bid, ask

    def spread_bps(self, symbol: str) -> float | None:
        with self.lock:
            return self.spread_bps_unlocked(symbol)

    def spread_bps_unlocked(self, symbol: str) -> float | None:
        bid, ask = self.best_bid_ask_unlocked(symbol)
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid * 10000.0 if mid > 0 else None

    def book_imbalance(self, symbol: str, levels: int = 5) -> float | None:
        with self.lock:
            return self.book_imbalance_unlocked(symbol, levels)

    def book_imbalance_unlocked(self, symbol: str, levels: int = 5) -> float | None:
        bids = self.top_bids_unlocked(symbol, levels)
        asks = self.top_asks_unlocked(symbol, levels)
        bid_notional = sum(p * q for p, q in bids)
        ask_notional = sum(p * q for p, q in asks)
        if bid_notional <= 0 or ask_notional <= 0:
            return None
        return bid_notional / ask_notional

    def book_is_fresh(self, symbol: str) -> bool:
        with self.lock:
            return self.book_is_fresh_unlocked(symbol)

    def book_is_fresh_unlocked(self, symbol: str) -> bool:
        if self.ohlc_only:
            return True
        if symbol not in self.book_symbols:
            return False
        b = self.books.get(symbol)
        if not b or not self.book_ready.get(symbol):
            return False
        age = self.now() - b.get("ts", 0.0)
        return 0.0 <= age <= self.cfg.book_stale_sec and bool(b.get("bids")) and bool(b.get("asks"))

    def book_quality_ok(self, symbol: str, max_spread_bps: float = 25.0, min_imbalance: float = 0.75, require_book: bool = True) -> bool:
        with self.lock:
            if self.ohlc_only:
                return True
            if symbol not in self.book_symbols:
                return not require_book
            if not self.book_is_fresh_unlocked(symbol):
                return False
            sp = self.spread_bps_unlocked(symbol)
            imb = self.book_imbalance_unlocked(symbol, 5)
            if sp is None or imb is None:
                return False
            return sp <= max_spread_bps and imb >= min_imbalance

    def mark_for_entry(self, symbol: str) -> float:
        with self.lock:
            if self.book_is_fresh_unlocked(symbol):
                _, ask = self.best_bid_ask_unlocked(symbol)
                if ask > 0:
                    return ask
            if symbol in self.last_price:
                return self.last_price[symbol]
            if self.state[symbol]:
                return self.state[symbol][-1][4]
            return 0.0

    def mark_for_exit(self, symbol: str) -> float:
        with self.lock:
            if self.book_is_fresh_unlocked(symbol):
                bid, _ = self.best_bid_ask_unlocked(symbol)
                if bid > 0:
                    return bid
            if symbol in self.last_price:
                return self.last_price[symbol]
            if self.state[symbol]:
                return self.state[symbol][-1][4]
            return 0.0

    def estimate_taker_buy_fill(self, symbol: str, quote_amount: float) -> tuple[float, float]:
        if quote_amount <= 0:
            return 0.0, 0.0
        with self.lock:
            asks = self.top_asks_unlocked(symbol, 10)
            if not asks:
                p = self.mark_for_entry(symbol)
                if p <= 0:
                    return 0.0, 0.0
                fill = p * (1 + self.cfg.slippage)
                return fill, quote_amount / fill

            remaining_quote = quote_amount
            spent = 0.0
            qty = 0.0
            for price, avail_qty in asks:
                level_quote = price * avail_qty
                take_quote = min(remaining_quote, level_quote)
                take_qty = take_quote / price
                spent += take_quote
                qty += take_qty
                remaining_quote -= take_quote
                if remaining_quote <= 1e-9:
                    break
            if remaining_quote > quote_amount * 0.01:
                return 0.0, 0.0
            avg = spent / qty if qty > 0 else 0.0
            return avg, qty

    def estimate_taker_sell_fill(self, symbol: str, base_qty: float) -> float:
        if base_qty <= 0:
            return 0.0
        with self.lock:
            bids = self.top_bids_unlocked(symbol, 10)
            if not bids:
                p = self.mark_for_exit(symbol)
                if p <= 0:
                    return 0.0
                return base_qty * p * (1 - self.cfg.slippage)

            remaining_qty = base_qty
            proceeds = 0.0
            for price, avail_qty in bids:
                take_qty = min(remaining_qty, avail_qty)
                proceeds += take_qty * price
                remaining_qty -= take_qty
                if remaining_qty <= 1e-12:
                    break
            if remaining_qty > base_qty * 0.01:
                return 0.0
            return proceeds

    def estimated_cost_floor(self, symbol: str) -> float:
        sp = self.spread_bps(symbol)
        spread_cost = (sp / 10000.0) if sp is not None else 0.002
        return 2.0 * self.cfg.fee_pct + spread_cost + 0.002

    # ── online Hawkes-like tape excitation ───────────────
    def _hawkes_impulse(self, notional: float) -> float:
        if not self.cfg.hawkes_enabled or notional <= 0:
            return 0.0
        return min(math.sqrt(notional), max(float(self.cfg.hawkes_impulse_cap), 1.0))

    def _hawkes_branching_unlocked(self) -> tuple[float, float, float]:
        """Return subcritical self/cross excitation coefficients.

        A Hawkes process is only stable when its branching ratio is below 1.
        Keep the runtime tunables permissive, then normalize them here so bad
        JSON/Optuna params cannot produce explosive intensities.
        """
        self_a = max(float(self.cfg.hawkes_self_excitation), 0.0)
        cross_a = max(float(self.cfg.hawkes_cross_excitation), 0.0)
        cap = max(0.01, min(float(self.cfg.hawkes_branching_cap), 0.99))
        total = self_a + cross_a
        if total > cap:
            scale = cap / max(total, 1e-12)
            self_a *= scale
            cross_a *= scale
            total = cap
        return self_a, cross_a, total

    def update_hawkes_unlocked(self, symbol: str, side: str | None, tnow: float, impulse: float = 0.0) -> None:
        if not self.cfg.hawkes_enabled:
            return

        tau = max(float(self.cfg.hawkes_decay_sec), 1e-6)
        st = self.hawkes_state[symbol]
        old_buy = float(st.get("buy", 0.0))
        old_sell = float(st.get("sell", 0.0))
        old_total = old_buy + old_sell
        last_ts = float(st.get("last_ts", 0.0))
        dt = max(0.0, tnow - last_ts) if last_ts > 0 else 0.0
        decay = math.exp(-dt / tau) if dt > 0 else 1.0
        buy = old_buy * decay
        sell = old_sell * decay

        if side == "buy":
            buy += impulse
        elif side == "sell":
            sell += impulse

        new_total = buy + sell
        if dt > 0:
            st["slope"] = (new_total - old_total) / max(dt, 1e-3)
        elif impulse > 0:
            st["slope"] = impulse

        st["buy"] = buy
        st["sell"] = sell
        st["last_ts"] = tnow

    def _hawkes_baseline_unlocked(self, symbol: str, tnow: float) -> tuple[float, float, float, float]:
        bdq = self.hawkes_base.get(symbol)
        baseline_duration = max(tnow - bdq[0][0], 1.0) if bdq and len(bdq) >= 2 else 0.0
        tau = max(float(self.cfg.hawkes_decay_sec), 1e-6)
        baseline_buy = 0.0
        baseline_sell = 0.0
        if baseline_duration > 0:
            baseline_buy = self.hawkes_base_buy_sum.get(symbol, 0.0) / baseline_duration * tau
            baseline_sell = self.hawkes_base_sell_sum.get(symbol, 0.0) / baseline_duration * tau
        baseline_total = baseline_buy + baseline_sell
        return baseline_buy, baseline_sell, baseline_total, baseline_duration

    @staticmethod
    def _hawkes_direction_from_ratio(ratio: float, buy_pressure: float, sell_pressure: float) -> float:
        if buy_pressure <= 0 and sell_pressure > 0:
            direction = -2.0
        elif sell_pressure <= 0 and buy_pressure > 0:
            direction = 2.0
        elif ratio > 0:
            direction = math.log(ratio)
        else:
            direction = 0.0
        return max(-2.0, min(2.0, direction))

    def prune_hawkes_base_unlocked(self, symbol: str, tnow: float) -> None:
        dq = self.hawkes_base[symbol]
        while dq and tnow - dq[0][0] > self.cfg.tape_baseline_sec:
            _, old_buy, old_sell = dq.popleft()
            self.hawkes_base_buy_sum[symbol] -= old_buy
            self.hawkes_base_sell_sum[symbol] -= old_sell

    def _hawkes_features_unlocked(
        self, symbol: str, tnow: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not self.cfg.hawkes_enabled:
            return {}, {}

        self.update_hawkes_unlocked(symbol, None, tnow, 0.0)
        self.prune_hawkes_base_unlocked(symbol, tnow)

        st = self.hawkes_state.get(symbol)
        if not st:
            return {}, {}

        buy = max(float(st.get("buy", 0.0)), 0.0)
        sell = max(float(st.get("sell", 0.0)), 0.0)
        total = buy + sell
        baseline_buy, baseline_sell, baseline_total, baseline_duration = self._hawkes_baseline_unlocked(symbol, tnow)
        self_a, cross_a, branching = self._hawkes_branching_unlocked()

        # Two-sided marked Hawkes intensity:
        #   lambda_buy  = mu_buy  + alpha_self * buy_trace  + alpha_cross * sell_trace
        #   lambda_sell = mu_sell + alpha_self * sell_trace + alpha_cross * buy_trace
        # The rolling baseline estimates the exogenous intensity integrated over
        # one decay horizon, while the decayed traces carry recent clustered flow.
        buy_intensity = baseline_buy + self_a * buy + cross_a * sell
        sell_intensity = baseline_sell + self_a * sell + cross_a * buy
        total_intensity = buy_intensity + sell_intensity

        ratio = buy_intensity / max(sell_intensity, 1e-9)
        raw_ratio = buy / max(sell, 1e-9)
        excitation = total_intensity / max(baseline_total, 1e-9) if baseline_total > 0 else 0.0
        raw_excitation = total / max(baseline_total, 1e-9) if baseline_total > 0 else 0.0
        slope = float(st.get("slope", 0.0))

        direction = self._hawkes_direction_from_ratio(ratio, buy_intensity, sell_intensity)
        magnitude = math.log1p(max(excitation - 1.0, 0.0))
        slope_norm = max(-1.0, min(1.0, slope / max(baseline_total, 1e-3) * 5.0))

        metrics = {
            "hawkes_buy": buy,
            "hawkes_sell": sell,
            "hawkes_total": total,
            "hawkes_buy_sell_ratio": ratio,
            "hawkes_raw_buy_sell_ratio": raw_ratio,
            "hawkes_sell_buy_ratio": sell_intensity / max(buy_intensity, 1e-9),
            "hawkes_raw_sell_buy_ratio": sell / max(buy, 1e-9),
            "hawkes_excitation": excitation,
            "hawkes_raw_excitation": raw_excitation,
            "hawkes_buy_excitation": buy_intensity / max(baseline_buy, 1e-9) if baseline_buy > 0 else 0.0,
            "hawkes_sell_excitation": sell_intensity / max(baseline_sell, 1e-9) if baseline_sell > 0 else 0.0,
            "hawkes_buy_intensity": buy_intensity,
            "hawkes_sell_intensity": sell_intensity,
            "hawkes_slope": slope,
            "hawkes_baseline_total": baseline_total,
            "hawkes_baseline_duration": baseline_duration,
            "hawkes_branching_ratio": branching,
        }
        features = {
            "direction": direction,
            "magnitude": magnitude,
            "slope_norm": slope_norm,
        }
        return metrics, features

    def _record_hawkes_history_unlocked(self, symbol: str, tnow: float, metrics: dict[str, float]) -> None:
        if not metrics:
            return
        if (
            metrics.get("hawkes_excitation", 0.0) <= 0.0
            and metrics.get("hawkes_buy", 0.0) <= 0.0
            and metrics.get("hawkes_sell", 0.0) <= 0.0
        ):
            return
        row = (
            tnow,
            float(metrics.get("hawkes_excitation", 0.0)),
            float(metrics.get("hawkes_buy_sell_ratio", 1.0)),
            float(metrics.get("hawkes_slope", 0.0)),
        )
        hist = self.hawkes_history[symbol]
        if hist and hist[-1][0] == tnow:
            hist[-1] = row
        else:
            hist.append(row)

    def _hawkes_prediction_value(
        self, params: dict[str, Any], direction: float, magnitude: float, slope_norm: float
    ) -> float:
        k = float(params["k"])
        sw = float(params["slope_weight"])
        b = float(params["bias"])
        pred = k * direction * magnitude * (1.0 + sw * slope_norm) + b
        return max(-0.02, min(0.02, pred))

    def _emit_hawkes_prediction_unlocked(
        self, symbol: str, tnow: float, features: dict[str, float]
    ) -> None:
        if not self.cfg.hawkes_enabled or not features:
            return
        st = self.hawkes_state[symbol]
        interval = max(float(self.cfg.hawkes_prediction_interval_sec), 0.0)
        last_pred_ts = float(st.get("last_pred_ts", 0.0))
        if last_pred_ts > 0 and tnow - last_pred_ts < interval:
            return

        direction = float(features.get("direction", 0.0))
        magnitude = float(features.get("magnitude", 0.0))
        slope_norm = float(features.get("slope_norm", 0.0))
        params = self.hawkes_learning[symbol]
        pred = self._hawkes_prediction_value(params, direction, magnitude, slope_norm)
        horizon = max(float(self.cfg.hawkes_prediction_horizon_sec), 1.0)
        self.hawkes_predictions[symbol].append((tnow, pred, horizon))

        # Store the exact feature vector that produced the delayed prediction.
        # The online SGD step must train against these features, not whatever
        # the Hawkes state looks like when the outcome matures.
        px_now = float(self.last_price.get(symbol, 0.0))
        if px_now > 0:
            self.hawkes_predictions_pending[symbol].append(
                (tnow, pred, px_now, horizon, direction, magnitude, slope_norm)
            )
        st["last_pred_ts"] = tnow

    def hawkes_snapshot_unlocked(self, symbol: str, tnow: float | None = None) -> dict[str, float]:
        if not self.cfg.hawkes_enabled:
            return {}

        tnow = self.now() if tnow is None else tnow
        metrics, _features = self._hawkes_features_unlocked(symbol, tnow)
        self._record_hawkes_history_unlocked(symbol, tnow, metrics)
        return metrics

    def _learn_matured_predictions_unlocked(
        self,
        symbol: str,
        tnow: float,
    ) -> None:
        """Pop matured (prediction, t0_price) entries; update params by gradient.

        Each matured entry stores the feature snapshot that produced the
        delayed prediction. The SGD step recomputes the prediction with the
        current parameters on those original features, which is the correct
        delayed online-learning update.

        Learning rate is intentionally tiny — 60s noise dominates a single
        sample, so we want the parameters to integrate across hundreds of
        observations.
        """
        pending = self.hawkes_predictions_pending.get(symbol)
        if not pending:
            return
        params = self.hawkes_learning[symbol]
        lr_k = 5e-5
        lr_sw = 5e-4
        lr_b = 1e-6

        while pending:
            item = pending[0]
            t0, pred_pct, px0, horizon = item[:4]
            if tnow - t0 < horizon:
                break
            pending.popleft()
            if len(item) >= 7:
                d, m, sn = float(item[4]), float(item[5]), float(item[6])
            else:
                # Compatibility with pending entries created before this
                # version; only the bias can learn without the original
                # feature snapshot.
                d, m, sn = 0.0, 0.0, 0.0

            px_mature = self._price_at_or_after_unlocked(symbol, t0 + horizon)
            if px0 <= 0 or px_mature <= 0:
                continue
            actual_pct = (px_mature / px0) - 1.0
            # Cap absurd outliers
            actual_pct = max(-0.05, min(0.05, actual_pct))
            pred_now = self._hawkes_prediction_value(params, d, m, sn)
            err = pred_now - actual_pct  # gradient of (pred - actual)^2 wrt pred is 2*err
            live_err = pred_pct - actual_pct

            # Gradient w.r.t. each parameter using the formula
            #   pred = k * d * m * (1 + sw * sn) + b
            # ∂pred/∂k  = d * m * (1 + sw * sn)
            # ∂pred/∂sw = k * d * m * sn
            # ∂pred/∂b  = 1
            sw_curr = float(params["slope_weight"])
            k_curr = float(params["k"])
            grad_k = 2.0 * err * d * m * (1.0 + sw_curr * sn)
            grad_sw = 2.0 * err * k_curr * d * m * sn
            grad_b = 2.0 * err

            new_k = k_curr - lr_k * grad_k
            new_sw = sw_curr - lr_sw * grad_sw
            new_b = float(params["bias"]) - lr_b * grad_b

            # Clamp to safe ranges
            new_k = max(1e-5, min(0.01, new_k))
            new_sw = max(-1.0, min(1.0, new_sw))
            new_b = max(-0.005, min(0.005, new_b))

            params["k"] = new_k
            params["slope_weight"] = new_sw
            params["bias"] = new_b
            params["last_pred_pct"] = pred_pct
            params["last_actual_pct"] = actual_pct

            n = int(params["n_matured"]) + 1
            params["n_matured"] = n
            # Directional hit (ignore samples where both are tiny)
            if abs(actual_pct) > 0.0002 and abs(pred_pct) > 0.0001:
                hit = 1 if (pred_pct >= 0) == (actual_pct >= 0) else 0
                params["hit_count"] = int(params["hit_count"]) + hit
                params["n_with_move"] = int(params.get("n_with_move", 0)) + 1
                # Rolling hit-rate over all matured samples with non-trivial moves.
                # We approximate with a slow EMA so it tracks recent quality.
                prev_rate = float(params["hit_rate"]) or 0.5
                alpha = 0.05
                params["hit_rate"] = (1 - alpha) * prev_rate + alpha * hit

            # Loss EMA (squared error)
            prev_loss = float(params["loss_ema"]) or live_err * live_err
            params["loss_ema"] = 0.95 * prev_loss + 0.05 * (live_err * live_err)

            self._reclassify_learning_status_unlocked(symbol)

    def _price_at_or_after_unlocked(self, symbol: str, target_ts: float) -> float:
        hist = self.price_history.get(symbol)
        if hist:
            for ts, px in hist:
                if ts >= target_ts and px > 0:
                    return float(px)
        return float(self.last_price.get(symbol, 0.0))

    def _reclassify_learning_status_unlocked(self, symbol: str) -> str:
        """Map (n_matured, hit_rate) to a status label.

        Status taxonomy (single source of truth — used by both the live
        update path and unit tests):

        - "cold":       n_matured < 50. Not enough samples to say anything.
        - "learning":   50 ≤ n_matured < 200. Still warming up.
        - "calibrated": n_matured ≥ 200 AND hit_rate ≥ 0.60. Has positive
                        directional edge. (Allowed to feed downstream gauges.)
        - "noisy":      n_matured ≥ 200 AND 0.45 ≤ hit_rate < 0.60. The
                        prediction has no real edge but isn't systematically
                        wrong either — noisy/random.
        - "inverted":   n_matured ≥ 200 AND hit_rate < 0.45. Systematically
                        wrong; if anything, the *inverse* could be useful.
        - "degraded":   was previously calibrated, hit_rate dropped to
                        [0.55, 0.60). Watching whether it recovers.

        Sticky cases:
        - Once "calibrated", a drop to [0.55, 0.60) goes to "degraded", not
          back to "noisy" — we want to flag transitions, not silently lose
          calibrated status to noise.
        - "degraded" returns to "calibrated" if hit_rate ≥ 0.60.
        """
        params = self.hawkes_learning[symbol]
        n = int(params.get("n_matured", 0))
        hr = float(params.get("hit_rate", 0.0))
        prev = str(params.get("status", "cold"))

        if n < 50:
            status = "cold"
        elif n < 200:
            status = "learning"
        elif hr >= 0.60:
            status = "calibrated"
        elif hr < 0.45:
            status = "inverted"
        elif prev == "calibrated" and 0.55 <= hr < 0.60:
            status = "degraded"
        elif prev == "degraded" and hr >= 0.60:
            status = "calibrated"
        elif prev == "degraded" and hr < 0.55:
            status = "noisy"
        else:
            status = "noisy"

        params["status"] = status
        return status

    def update_inertia_unlocked(
        self,
        symbol: str,
        name: str,
        value: float,
        pos_threshold: float,
        neg_threshold: float,
        confirm_ticks: int = 4,
    ) -> int:
        key = (symbol, name)
        tr = self.inertia.get(key)
        if tr is None:
            tr = self._inertia_cls(confirm_ticks=confirm_ticks)
            self.inertia[key] = tr
        return tr.update(value, pos_threshold, neg_threshold)

    def get_inertia_state(self, symbol: str, name: str) -> int:
        tr = self.inertia.get((symbol, name))
        return tr.state if tr is not None else 0

    # ── tape helpers ────────────────────────────────────
    def add_trade_to_tape(self, symbol: str, side: str, price: float, qty: float) -> None:
        tnow = self.now()
        notional = price * qty
        hawkes_impulse = self._hawkes_impulse(notional)
        with self.lock:
            self.last_price[symbol] = price
            dq = self.tape[symbol]
            dq.append((tnow, side, price, qty, notional))
            if side == "buy":
                self.tape_buy_sum[symbol] += notional
                buy_not = notional
                sell_not = 0.0
                hawkes_buy = hawkes_impulse
                hawkes_sell = 0.0
            else:
                self.tape_sell_sum[symbol] += notional
                buy_not = 0.0
                sell_not = notional
                hawkes_buy = 0.0
                hawkes_sell = hawkes_impulse
            self.base_tape[symbol].append((tnow, buy_not, sell_not, notional))
            self.base_buy_sum[symbol] += buy_not
            self.base_total_sum[symbol] += notional
            self.hawkes_base[symbol].append((tnow, hawkes_buy, hawkes_sell))
            self.hawkes_base_buy_sum[symbol] += hawkes_buy
            self.hawkes_base_sell_sum[symbol] += hawkes_sell
            self.update_hawkes_unlocked(symbol, side, tnow, hawkes_impulse)
            self.prune_hawkes_base_unlocked(symbol, tnow)
            self.prune_tape_unlocked(symbol, tnow)
            self.price_history[symbol].append((tnow, price))
            metrics, features = self._hawkes_features_unlocked(symbol, tnow)
            self._record_hawkes_history_unlocked(symbol, tnow, metrics)
            self._learn_matured_predictions_unlocked(symbol, tnow)
            self._emit_hawkes_prediction_unlocked(symbol, tnow, features)
            self.tape_pressure_history[symbol].append((tnow, self.tape_buy_sum[symbol], self.tape_sell_sum[symbol]))

    def prune_tape(self, symbol: str, tnow: float | None = None) -> None:
        with self.lock:
            self.prune_tape_unlocked(symbol, tnow or self.now())

    def prune_tape_unlocked(self, symbol: str, tnow: float) -> None:
        dq = self.tape[symbol]
        while dq and tnow - dq[0][0] > self.cfg.tape_window_sec:
            _, old_side, _, _, old_notional = dq.popleft()
            if old_side == "buy":
                self.tape_buy_sum[symbol] -= old_notional
            else:
                self.tape_sell_sum[symbol] -= old_notional
        bdq = self.base_tape[symbol]
        while bdq and tnow - bdq[0][0] > self.cfg.tape_baseline_sec:
            _, old_buy, _, old_total = bdq.popleft()
            self.base_buy_sum[symbol] -= old_buy
            self.base_total_sum[symbol] -= old_total
        self.prune_hawkes_base_unlocked(symbol, tnow)

    def recalc_running_sums(self) -> None:
        with self.lock:
            for sym in list(self.tape.keys()):
                self.tape_buy_sum[sym] = sum(x[4] for x in self.tape[sym] if x[1] == "buy")
                self.tape_sell_sum[sym] = sum(x[4] for x in self.tape[sym] if x[1] == "sell")
            for sym in list(self.base_tape.keys()):
                self.base_buy_sum[sym] = sum(x[1] for x in self.base_tape[sym])
                self.base_total_sum[sym] = sum(x[3] for x in self.base_tape[sym])
            for sym in list(self.hawkes_base.keys()):
                self.hawkes_base_buy_sum[sym] = sum(x[1] for x in self.hawkes_base[sym])
                self.hawkes_base_sell_sum[sym] = sum(x[2] for x in self.hawkes_base[sym])

    def has_recent_tape(self, symbol: str, max_age_sec: float | None = None) -> bool:
        with self.lock:
            dq = self.tape.get(symbol)
            if not dq:
                return False
            if max_age_sec is None:
                max_age_sec = self.cfg.tape_window_sec + 1.0
            return self.now() - dq[-1][0] <= max_age_sec

    def live_microstructure_ok(
        self,
        symbol: str,
        min_trades: int | None = None,
        min_notional: float | None = None,
        min_buy_share: float | None = None,
        max_spread_bps: float | None = None,
        min_book_imbalance: float | None = None,
    ) -> bool:
        if self.ohlc_only:
            return True
        cfg = self.cfg
        min_trades = cfg.macro_min_tape_trades if min_trades is None else min_trades
        min_notional = cfg.macro_min_tape_notional if min_notional is None else min_notional
        min_buy_share = cfg.macro_min_tape_buy_share if min_buy_share is None else min_buy_share
        max_spread_bps = cfg.macro_max_spread_bps if max_spread_bps is None else max_spread_bps
        min_book_imbalance = cfg.macro_min_book_imbalance if min_book_imbalance is None else min_book_imbalance

        with self.lock:
            if symbol not in self.trade_symbols or symbol not in self.book_symbols:
                return False
            if not self.book_quality_ok(symbol, max_spread_bps=max_spread_bps, min_imbalance=min_book_imbalance, require_book=True):
                return False
            self.prune_tape_unlocked(symbol, self.now())
            dq = self.tape.get(symbol)
            if not dq or len(dq) < min_trades:
                return False
            buy_not = self.tape_buy_sum.get(symbol, 0.0)
            sell_not = self.tape_sell_sum.get(symbol, 0.0)
            total = buy_not + sell_not
            if total < min_notional:
                return False
            return buy_not / max(total, 1.0) >= min_buy_share

    # ── context / snapshots ─────────────────────────────
    def btc_context_ok(self, strict: bool = False) -> bool:
        with self.lock:
            hist = self.closed_state[BTC_SYMBOL]
            if len(hist) < 8:
                return True
            c = hist[-1][4]
            r5 = c / hist[-6][4] - 1.0 if len(hist) >= 6 and hist[-6][4] > 0 else 0.0
            r15 = c / hist[-16][4] - 1.0 if len(hist) >= 16 and hist[-16][4] > 0 else 0.0
            if strict:
                return r5 > -0.004 and r15 > -0.010
            return r5 > -0.008 and r15 > -0.018

    def btc_flush_active(self) -> bool:
        with self.lock:
            hist = self.closed_state[BTC_SYMBOL]
            if len(hist) < 3:
                return False
            c = hist[-1][4]
            r1 = c / hist[-2][4] - 1.0 if hist[-2][4] > 0 else 0.0
            r3 = c / hist[-4][4] - 1.0 if len(hist) >= 4 and hist[-4][4] > 0 else 0.0
            return r1 < -0.012 or r3 < -0.020

    def snapshot_micro(self, symbol: str) -> dict[str, Any] | None:
        with self.lock:
            return self.snapshot_micro_unlocked(symbol)

    def snapshot_micro_unlocked(self, symbol: str) -> dict[str, Any] | None:
        tnow = self.now()
        dq = self.tape.get(symbol)
        if not dq:
            return None
        buy_not = self.tape_buy_sum.get(symbol, 0.0)
        sell_not = self.tape_sell_sum.get(symbol, 0.0)
        bdq = self.base_tape.get(symbol)
        baseline_buy = 0.0
        baseline_total = 0.0
        baseline_duration = 0.0
        baseline_buy_share = 0.5
        if bdq and len(bdq) >= 2:
            baseline_duration = max(tnow - bdq[0][0], 1.0)
            baseline_buy = self.base_buy_sum.get(symbol, 0.0) * (self.cfg.tape_window_sec / baseline_duration)
            baseline_total = self.base_total_sum.get(symbol, 0.0) * (self.cfg.tape_window_sec / baseline_duration)
            raw_total = self.base_total_sum.get(symbol, 0.0)
            if raw_total > 0:
                baseline_buy_share = self.base_buy_sum.get(symbol, 0.0) / raw_total
        burst_ratio = (buy_not / baseline_buy) if baseline_buy > 0 else 0.0
        imbalance = buy_not / max(sell_not, 1.0)
        first_price = dq[0][2] if dq else 0.0
        last = dq[-1][2] if dq else 0.0
        move_pct = (last / first_price - 1.0) if first_price > 0 else 0.0
        sp = self.spread_bps_unlocked(symbol)
        book_imb = self.book_imbalance_unlocked(symbol, 5)
        max_spread = 10.0 if symbol in ("BTC/USD", "ETH/USD") else 150.0
        base = {
            "trades": len(dq),
            "buy_not": buy_not,
            "sell_not": sell_not,
            "burst_ratio": burst_ratio,
            "imbalance": imbalance,
            "move_pct": move_pct,
            "spread_bps": sp,
            "book_imb": book_imb,
            "max_spread": max_spread,
            "baseline_total_per_window": baseline_total,
            "baseline_duration": baseline_duration,
            "baseline_buy_share": baseline_buy_share,
        }
        base.update(self.hawkes_snapshot_unlocked(symbol, tnow))
        base.update(self.flow_snapshot_unlocked(symbol))
        return base

    def snapshot_thrust(self, symbol: str) -> dict[str, Any] | None:
        with self.lock:
            return self.snapshot_thrust_unlocked(symbol)

    def snapshot_thrust_unlocked(self, symbol: str) -> dict[str, Any] | None:
        hist = self.closed_state.get(symbol)
        if not hist or len(hist) < self.cfg.thrust_lookback + 1:
            return None
        c = hist[-1]
        close = c[4]
        if close <= 0:
            return None
        r3 = close / hist[-4][4] - 1.0 if len(hist) >= 4 and hist[-4][4] > 0 else 0.0
        r5 = close / hist[-6][4] - 1.0 if len(hist) >= 6 and hist[-6][4] > 0 else 0.0
        r15 = close / hist[-16][4] - 1.0 if len(hist) >= 16 and hist[-16][4] > 0 else 0.0
        prev = list(hist)[-(self.cfg.thrust_lookback + 1):-1]
        med_vol = median([x[5] for x in prev], 0.0)
        vol_x = c[5] / med_vol if med_vol > 0 else 0.0
        cp = close_position(c)
        return {"r3": r3, "r5": r5, "r15": r15, "vol_x": vol_x, "close_pos": cp}

    def remember_micro_snapshot(self, symbol: str, snap: dict[str, Any]) -> None:
        with self.lock:
            self.decision_snapshots[symbol] = dict(snap)

    def update_decision_snapshot(self, symbol: str, values: dict[str, Any]) -> None:
        with self.lock:
            self.decision_snapshots.setdefault(symbol, {}).update(values)

    def mark_entry(self, when: float, pair: str, price: float, regime: str) -> None:
        with self.lock:
            self.entry_marks.append((when, pair, price, regime))

    def mark_exit(self, when: float, pair: str, price: float, reason: str, pnl: float) -> None:
        with self.lock:
            self.exit_marks.append((when, pair, price, reason, pnl))

    def sample_capital(self, when: float, cap: float) -> None:
        with self.lock:
            self.capital_history.append((when, cap))

    def sample_performance(self, when: float, perf: dict[str, Any]) -> None:
        with self.lock:
            self.performance_history.append((
                when,
                float(perf.get("net_return") or 0.0),
                float(perf.get("return_per_hour") or 0.0),
                float(perf.get("return_per_exposure_hour") or 0.0),
                float(perf.get("current_drawdown") or 0.0),
                float(perf.get("horizon_score") or 0.0),
                float(perf.get("avg_hold_sec") or 0.0),
                float(perf.get("exposure_ratio") or 0.0),
                float(perf.get("avg_return_velocity_pct_per_min") or 0.0),
                float(perf.get("trades") or 0.0),
            ))

    def flow_snapshot_unlocked(self, symbol: str) -> dict[str, float]:
        dq = list(self.tape.get(symbol, []))
        if not dq:
            return {}
        buy_not = self.tape_buy_sum.get(symbol, 0.0)
        sell_not = self.tape_sell_sum.get(symbol, 0.0)
        bid_depth = sum(p * q for p, q in self.top_bids_unlocked(symbol, 10))
        ask_depth = sum(p * q for p, q in self.top_asks_unlocked(symbol, 10))
        return flow_metrics.compute(
            dq,
            tnow=self.now(),
            buy_not=buy_not,
            sell_not=sell_not,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth,
        )

    def flow_snapshot(self, symbol: str) -> dict[str, float]:
        with self.lock:
            return self.flow_snapshot_unlocked(symbol)

    def focus_candidate(self, preferred: str | None, position_pair: str | None = None) -> str:
        with self.lock:
            if preferred:
                return preferred
            if position_pair:
                return position_pair
            best, best_n = None, 0
            for sym, dq in self.tape.items():
                if len(dq) > best_n:
                    best, best_n = sym, len(dq)
            return best or BTC_SYMBOL

    def dashboard_view(self, focus: str, symbols: list[str] | None = None) -> MarketView:
        with self.lock:
            snap = self.decision_snapshots.get(focus, {})
            micro = dict(snap)
            fd = self.flow_snapshot_unlocked(focus)
            micro.update(fd)
            tref = self.now()
            self.flow_history[focus].append(
                (
                    tref,
                    float(fd.get("flow_churn", 0.0)),
                    float(fd.get("flow_viscosity", 0.0)),
                    float(fd.get("flow_turbulence_ret_var", 0.0)),
                    float(fd.get("flow_price_accel", 0.0)),
                    float(fd.get("flow_signed_usd_per_s", 0.0)),
                )
            )
            thrust = {k: v for k, v in snap.items() if k.startswith("thrust_")}
            chart_symbols = list(symbols) if symbols else [focus]
            multi_prices: dict[str, tuple[tuple[float, float], ...]] = {}
            multi_hawkes_history: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
            multi_hawkes_predictions: dict[str, tuple[tuple[float, float, float], ...]] = {}
            hawkes_learning_view: dict[str, dict[str, Any]] = {}
            inertia_state_view: dict[str, dict[str, int]] = {}
            for sym in chart_symbols:
                # Make sure each chart symbol has a current Hawkes snapshot so
                # per-symbol history / learning fires even before the trader
                # has touched the symbol this tick. The call is idempotent —
                # it appends one history entry per invocation, no side effects
                # on trading.
                if self.cfg.hawkes_enabled:
                    self.hawkes_snapshot_unlocked(sym, tref)
                series = self.price_history.get(sym)
                if series:
                    multi_prices[sym] = tuple(series)
                hh = self.hawkes_history.get(sym)
                if hh:
                    multi_hawkes_history[sym] = tuple(hh)
                hp = self.hawkes_predictions.get(sym)
                if hp:
                    multi_hawkes_predictions[sym] = tuple(hp)
                if sym in self.hawkes_learning:
                    hawkes_learning_view[sym] = dict(self.hawkes_learning[sym])

                # Update per-symbol inertia signals from the latest Hawkes snapshot.
                # "pressure": derived from log(buy/sell ratio) → state ∈ {-1,0,+1}
                # "readiness": derived from excitation*direction → state ∈ {-1,0,+1}
                latest = hh[-1] if hh else None
                if latest is not None:
                    _, exc, ratio_h, _ = latest
                    pressure_val = math.log(max(ratio_h, 1e-9))
                    self.update_inertia_unlocked(
                        sym, "pressure", pressure_val,
                        pos_threshold=0.15, neg_threshold=-0.15, confirm_ticks=4,
                    )
                    readiness_val = (exc - 1.0) * pressure_val
                    self.update_inertia_unlocked(
                        sym, "readiness", readiness_val,
                        pos_threshold=0.20, neg_threshold=-0.20, confirm_ticks=5,
                    )
                inertia_state_view[sym] = {
                    "pressure": self.get_inertia_state(sym, "pressure"),
                    "readiness": self.get_inertia_state(sym, "readiness"),
                }

            # BTC backdrop inertia: use 5m return as the raw signal, with
            # asymmetric thresholds so a flush is sticky-bearish quickly but
            # bullish takes sustained recovery.
            backdrop_raw = 0.0
            hist = self.closed_state.get(BTC_SYMBOL)
            if hist and len(hist) >= 6:
                c = hist[-1][4]
                if c > 0 and hist[-6][4] > 0:
                    backdrop_raw = c / hist[-6][4] - 1.0
            btc_state = self.btc_backdrop_inertia.update(
                backdrop_raw, pos_threshold=0.0005, neg_threshold=-0.004
            )

            return MarketView(
                ts=self.now(),
                focus=focus,
                price_series=tuple(self.price_history.get(focus, ())),
                tape_series=tuple(self.tape_pressure_history.get(focus, ())),
                bids=tuple(self.top_bids_unlocked(focus, 10)),
                asks=tuple(self.top_asks_unlocked(focus, 10)),
                entry_marks=tuple(self.entry_marks),
                exit_marks=tuple(self.exit_marks),
                capital_history=tuple(self.capital_history),
                micro=micro,
                thrust=thrust,
                last_price=self.last_price.get(focus, 0.0),
                spread_bps=self.spread_bps_unlocked(focus),
                book_imbalance=self.book_imbalance_unlocked(focus, 5),
                flow_series=tuple(self.flow_history[focus]),
                performance_history=tuple(self.performance_history),
                multi_prices=multi_prices,
                hawkes_history=tuple(self.hawkes_history.get(focus, ())),
                hawkes_predictions=tuple(self.hawkes_predictions.get(focus, ())),
                multi_hawkes_history=multi_hawkes_history,
                multi_hawkes_predictions=multi_hawkes_predictions,
                hawkes_learning=hawkes_learning_view,
                inertia_state=inertia_state_view,
                btc_backdrop_state=btc_state,
            )

    def clone_shallow_data(self) -> dict[str, Any]:
        """Debug/test helper; not used by Optuna anymore."""
        with self.lock:
            return {
                "state": {k: list(v) for k, v in self.state.items()},
                "closed_state": {k: list(v) for k, v in self.closed_state.items()},
                "last_price": dict(self.last_price),
                "books": copy.deepcopy(self.books),
            }
