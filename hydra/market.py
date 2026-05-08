from __future__ import annotations

import copy
import dataclasses
import threading
from collections import defaultdict, deque
from typing import Any

import numpy as np

from .config import Config
from .constants import (
    BOOK_DEPTH,
    BOOK_STALE_SEC,
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
    # (ts, net_ret, ret_per_hour, ret_per_exposure_hour, drawdown, horizon_score, avg_hold_sec, exposure_ratio, avg_velocity_pct_min, trades)
    performance_history: tuple[tuple[float, float, float, float, float, float, float, float, float, float], ...]


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
        return 0.0 <= age <= BOOK_STALE_SEC and bool(b.get("bids")) and bool(b.get("asks"))

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

    # ── tape helpers ────────────────────────────────────
    def add_trade_to_tape(self, symbol: str, side: str, price: float, qty: float) -> None:
        tnow = self.now()
        notional = price * qty
        with self.lock:
            self.last_price[symbol] = price
            dq = self.tape[symbol]
            dq.append((tnow, side, price, qty, notional))
            if side == "buy":
                self.tape_buy_sum[symbol] += notional
                buy_not = notional
                sell_not = 0.0
            else:
                self.tape_sell_sum[symbol] += notional
                buy_not = 0.0
                sell_not = notional
            self.base_tape[symbol].append((tnow, buy_not, sell_not, notional))
            self.base_buy_sum[symbol] += buy_not
            self.base_total_sum[symbol] += notional
            self.prune_tape_unlocked(symbol, tnow)
            self.price_history[symbol].append((tnow, price))
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

    def recalc_running_sums(self) -> None:
        with self.lock:
            for sym in list(self.tape.keys()):
                self.tape_buy_sum[sym] = sum(x[4] for x in self.tape[sym] if x[1] == "buy")
                self.tape_sell_sum[sym] = sum(x[4] for x in self.tape[sym] if x[1] == "sell")
            for sym in list(self.base_tape.keys()):
                self.base_buy_sum[sym] = sum(x[1] for x in self.base_tape[sym])
                self.base_total_sum[sym] = sum(x[3] for x in self.base_tape[sym])

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
                float(perf.get("net_return", 0.0) or 0.0),
                float(perf.get("return_per_hour", 0.0) or 0.0),
                float(perf.get("return_per_exposure_hour", 0.0) or 0.0),
                float(perf.get("current_drawdown", 0.0) or 0.0),
                float(perf.get("horizon_score", 0.0) or 0.0),
                float(perf.get("avg_hold_sec", 0.0) or 0.0),
                float(perf.get("exposure_ratio", 0.0) or 0.0),
                float(perf.get("avg_return_velocity_pct_per_min", 0.0) or 0.0),
                float(perf.get("trades", 0.0) or 0.0),
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

    def dashboard_view(self, focus: str) -> MarketView:
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



