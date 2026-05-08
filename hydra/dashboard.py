from __future__ import annotations

import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from .constants import BTC_SYMBOL, DASH_PRICE_LEN
from .flow_metrics import format_lines
from .utils import breakeven_move_pct

_EMPTY_OFFSETS = np.empty((0, 2))
_EMPTY_POLY = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

MICRO_LABELS = ["trades", "buy notional", "tape imb", "burst x", "move %", "book imb", "spread bps"]
MICRO_FMT = ["{:.0f}", "${:,.0f}", "{:.2f}", "{:.2f}x", "{:.2f}%", "{:.2f}", "{:.1f}"]
MICRO_GREATER = [True, True, True, True, True, True, False]

THRUST_LABELS = ["r3 (3m)", "r5 (5m)", "r15 (15m)", "vol x", "close pos"]
THRUST_FMT = ["{:.2f}%", "{:.2f}%", "{:+.2f}%", "{:.2f}x", "{:.2f}"]
THRUST_GREATER = [True, True, True, True, True]


def _set_axis_plain_numbers(ax):
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def _set_poly_between(collection: PolyCollection, xs, y0, ys):
    if not xs or not ys:
        collection.set_verts([_EMPTY_POLY])
        return
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if np.isscalar(y0):
        y0s = np.full_like(xs, float(y0), dtype=float)
    else:
        y0s = np.asarray(y0, dtype=float)
    verts = np.column_stack([np.r_[xs, xs[::-1]], np.r_[ys, y0s[::-1]]])
    collection.set_verts([verts])


def _pad_limits(vals, frac=0.06, fallback=(0.0, 1.0)):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return fallback
    lo, hi = min(vals), max(vals)
    if lo == hi:
        pad = abs(lo) * frac or 1.0
    else:
        pad = (hi - lo) * frac
    return lo - pad, hi + pad


class Dashboard:
    """Persistent-artist Matplotlib dashboard.

    The update callback only consumes an immutable snapshot from HydraEngine.
    It does not iterate or mutate live market dictionaries/deques.
    """

    HELP_LINES = [
        "1/2/3 = lock focus on BTC/ETH/SOL  |  0 = auto",
        "o = start/stop Optuna on closed recent recordings",
        "r = list available recordings in console",
        "? = print this help",
    ]

    def __init__(self, engine, cfg, params_source: str = "defaults", optuna_controller=None, title_suffix: str = ""):
        plt.style.use("dark_background")
        plt.rcParams["text.parse_math"] = False
        self.engine = engine
        self.cfg = cfg
        self.params_source = params_source
        self.optuna = optuna_controller
        self.locked_focus: str | None = None
        self.focus = BTC_SYMBOL

        self.fig = plt.figure(figsize=(17, 12))
        self.fig.canvas.manager.set_window_title(f"Hydra Trader — LIVE [{params_source}]{title_suffix}")
        gs = GridSpec(
            6, 4, figure=self.fig,
            height_ratios=[2.0, 1.9, 1.35, 1.0, 1.15, 1.25],
            width_ratios=[1.5, 1.0, 1.5, 1.0],
            left=0.06, right=0.985, top=0.955, bottom=0.040,
            hspace=0.58, wspace=0.42,
        )
        self.ax_price = self.fig.add_subplot(gs[0, 0:2])
        self.ax_capital = self.fig.add_subplot(gs[0, 2:4])
        self.ax_book = self.fig.add_subplot(gs[1, 0])
        self.ax_tape = self.fig.add_subplot(gs[1, 1:3])
        self.ax_micro = self.fig.add_subplot(gs[2, 0:2])
        self.ax_thrust = self.fig.add_subplot(gs[2, 2:4])
        self.ax_flow = self.fig.add_subplot(gs[3, :])
        self.ax_performance = self.fig.add_subplot(gs[4, :])
        self.ax_status = self.fig.add_subplot(gs[5, 0:2])
        self.ax_trades = self.fig.add_subplot(gs[5, 2:4])
        self.ax_regime = self.fig.add_subplot(gs[1, 3])
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._setup_price()
        self._setup_capital()
        self._setup_book()
        self._setup_tape()
        self._setup_flow()
        self._setup_performance()
        self._setup_gauge(self.ax_micro, "_micro", MICRO_LABELS, MICRO_FMT, "book_ignition")
        self._setup_gauge(self.ax_thrust, "_thrust", THRUST_LABELS, THRUST_FMT, "macro_thrust")
        self._setup_text_panel(self.ax_status, "_status", "Strategy state")
        self._setup_text_panel(self.ax_trades, "_trades", "Recent trades", mono=True)
        self._setup_text_panel(self.ax_regime, "_regime", "Context", mono=True, fontsize=7.5)

    def on_key(self, event):
        k = event.key
        if k == "1":
            self.locked_focus = "BTC/USD"
        elif k == "2":
            self.locked_focus = "ETH/USD"
        elif k == "3":
            self.locked_focus = "SOL/USD"
        elif k == "0":
            self.locked_focus = None
        elif k == "o":
            self.toggle_optuna()
            return
        elif k == "r":
            from .data import list_recordings
            recs = list_recordings()
            if not recs:
                print("[DASH] no recordings found in ./runs/ or .")
            else:
                print("[DASH] recordings (newest first):")
                for p in recs:
                    age_min = (time.time() - __import__('os').path.getmtime(p)) / 60.0
                    sz = __import__('os').path.getsize(p) / 1024.0
                    print(f"   {p}  ({sz:.1f} KB, {age_min:.1f} min ago)")
            return
        elif k == "?":
            print("\n[DASH] keybindings:")
            for line in self.HELP_LINES:
                print("   " + line)
            return
        else:
            return
        print(f"  [DASH] focus locked to: {self.locked_focus or 'AUTO'}")

    def toggle_optuna(self):
        if self.optuna is None:
            print("[DASH] optuna controller not configured")
            return
        state = self.optuna.snapshot()
        if state.get("running"):
            self.optuna.stop()
            print("[DASH] optuna: stopping")
            return
        ok, info = self.optuna.start(trigger_note="manual (key 'o')")
        print(f"[DASH] optuna: {'started on ' + info if ok else info}")

    @staticmethod
    def _style_axes(ax, grid=True, alpha=0.15):
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(axis="both", labelsize=8, colors="#aaaaaa")
        if grid:
            ax.grid(True, alpha=alpha)
        _set_axis_plain_numbers(ax)

    def _setup_price(self):
        ax = self.ax_price
        ax.set_xlabel("seconds ago", fontsize=8, color="#999")
        self._style_axes(ax)
        self.price_line, = ax.plot(
            [], [], color="#3aa6ff", linewidth=1.4, drawstyle="steps-post"
        )
        self.last_px_hline = ax.axhline(0, color="#3aa6ff", linewidth=0.5, alpha=0.4, linestyle="--")
        self.anchor_hline = ax.axhline(0, color="#f5b942", linewidth=0.8, alpha=0.7, linestyle=":")
        self.entry_hline = ax.axhline(0, color="#48d597", linewidth=0.8, alpha=0.7, linestyle="--")
        self.trail_hline = ax.axhline(0, color="#ff66cc", linewidth=0.8, alpha=0.7, linestyle="--")
        self.anchor_text = ax.text(0, 0, "", color="#f5b942", fontsize=8, va="bottom")
        self.entry_text = ax.text(0, 0, "", color="#48d597", fontsize=8, va="bottom")
        self.trail_text = ax.text(0, 0, "", color="#ff66cc", fontsize=8, va="top")
        self.entry_scatter = ax.scatter([], [], marker="^", color="#48d597", s=80, zorder=5, edgecolors="white", linewidths=0.5)
        self.exit_pos_scatter = ax.scatter([], [], marker="v", color="#48d597", s=80, zorder=5, edgecolors="white", linewidths=0.5)
        self.exit_neg_scatter = ax.scatter([], [], marker="v", color="#ff5555", s=80, zorder=5, edgecolors="white", linewidths=0.5)
        self.price_title = ax.set_title("", color="#ffffff", fontsize=11, loc="left")
        for art in (self.last_px_hline, self.anchor_hline, self.entry_hline, self.trail_hline, self.anchor_text, self.entry_text, self.trail_text):
            art.set_visible(False)

    @staticmethod
    def _set_offsets_safe(scatter, points):
        scatter.set_offsets(np.asarray(points, dtype=float) if points else _EMPTY_OFFSETS)

    def draw_price(self, snap):
        ax = self.ax_price
        view = snap["market"]
        data = list(view.price_series)
        sym = snap["focus"]
        if not data:
            self.price_line.set_data([], [])
            for art in (self.last_px_hline, self.anchor_hline, self.entry_hline, self.trail_hline, self.anchor_text, self.entry_text, self.trail_text):
                art.set_visible(False)
            self._set_offsets_safe(self.entry_scatter, [])
            self._set_offsets_safe(self.exit_pos_scatter, [])
            self._set_offsets_safe(self.exit_neg_scatter, [])
            self.price_title.set_text(f"{sym} — waiting for data")
            return
        n = view.ts
        xs = [(t - n) for t, _ in data]
        ys = [p for _, p in data]
        last_px = ys[-1]
        self.price_line.set_data(xs, ys)
        self.last_px_hline.set_ydata([last_px, last_px])
        self.last_px_hline.set_visible(True)

        watch = dict(snap["watch"]).get(sym) if snap.get("watch") else None
        if watch:
            self.anchor_hline.set_ydata([watch["anchor"], watch["anchor"]])
            self.anchor_hline.set_visible(True)
            self.anchor_text.set_position((xs[0], watch["anchor"]))
            self.anchor_text.set_text(f"  anchor {watch['anchor']:.4g}")
            self.anchor_text.set_visible(True)
        else:
            self.anchor_hline.set_visible(False)
            self.anchor_text.set_visible(False)

        pos = snap.get("position")
        if pos and pos["pair"] == sym:
            entry = pos["entry"]
            self.entry_hline.set_ydata([entry, entry])
            self.entry_hline.set_visible(True)
            self.entry_text.set_position((xs[0], entry))
            self.entry_text.set_text(f"  entry {entry:.4g}")
            self.entry_text.set_visible(True)
            r = self.cfg.risk(pos["regime"])
            if pos.get("trail_active"):
                trail = pos["peak_mark"] * (1 - r["trail_pct"])
                self.trail_hline.set_ydata([trail, trail])
                self.trail_hline.set_visible(True)
                self.trail_text.set_position((xs[0], trail))
                self.trail_text.set_text(f"  trail {trail:.4g}")
                self.trail_text.set_visible(True)
            else:
                self.trail_hline.set_visible(False)
                self.trail_text.set_visible(False)
        else:
            self.entry_hline.set_visible(False)
            self.entry_text.set_visible(False)
            self.trail_hline.set_visible(False)
            self.trail_text.set_visible(False)

        win_start = n - DASH_PRICE_LEN
        e_pts, xp_pts, xn_pts = [], [], []
        for et, esym, ep, _ in view.entry_marks:
            if esym == sym and et >= win_start:
                e_pts.append((et - n, ep))
        for et, esym, ep, _, pnl in view.exit_marks:
            if esym != sym or et < win_start:
                continue
            (xp_pts if pnl > 0 else xn_pts).append((et - n, ep))
        self._set_offsets_safe(self.entry_scatter, e_pts)
        self._set_offsets_safe(self.exit_pos_scatter, xp_pts)
        self._set_offsets_safe(self.exit_neg_scatter, xn_pts)

        title = f"{sym}  last {last_px:.6g}"
        if view.spread_bps is not None:
            title += f"  sp {view.spread_bps:.1f}bps"
        if view.book_imbalance is not None:
            title += f"  book {view.book_imbalance:.2f}"
        self.price_title.set_text(title)
        ax.set_xlim(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
        ax.set_ylim(*_pad_limits(ys + [last_px]))

    def _setup_capital(self):
        ax = self.ax_capital
        ax.set_xlabel("minutes ago", fontsize=8, color="#999")
        self._style_axes(ax)
        self.cap_line, = ax.plot(
            [], [], color="#48d597", linewidth=1.6, drawstyle="steps-post"
        )
        self.cap_baseline = ax.axhline(self.cfg.start_capital, color="#888", linewidth=0.7, linestyle="--", alpha=0.7)
        self.cap_fill = PolyCollection([_EMPTY_POLY], facecolors="#48d597", alpha=0.18, edgecolors="none")
        ax.add_collection(self.cap_fill)
        self.cap_title = ax.set_title("Capital — waiting", color="#cccccc", fontsize=10, loc="left")

    def draw_capital(self, snap):
        ax = self.ax_capital
        data = list(snap["market"].capital_history)
        if not data:
            self.cap_line.set_data([], [])
            self.cap_title.set_text("Capital — waiting")
            return
        n = snap["market"].ts
        xs = [(t - n) / 60.0 for t, _ in data]
        ys = [c for _, c in data]
        cur = ys[-1]
        color = "#48d597" if cur >= self.cfg.start_capital else "#ff5555"
        self.cap_line.set_data(xs, ys)
        self.cap_line.set_color(color)
        self.cap_baseline.set_ydata([self.cfg.start_capital, self.cfg.start_capital])
        self.cap_fill.set_facecolor(color)
        _set_poly_between(self.cap_fill, xs, self.cfg.start_capital, ys)
        net = cur / self.cfg.start_capital - 1.0
        self.cap_title.set_text(f"Capital ${cur:.4f}  ({net*100:+.2f}% from ${self.cfg.start_capital:.2f})  trades {len(snap['trades'])}")
        ax.set_xlim(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
        ax.set_ylim(*_pad_limits(ys + [self.cfg.start_capital]))

    def _setup_book(self):
        ax = self.ax_book
        self._style_axes(ax, alpha=0.15)
        ax.grid(True, alpha=0.15, axis="x")
        self.book_bid_rects = [Rectangle((0, 0), 0, 0, color="#48d597", alpha=0.7) for _ in range(10)]
        self.book_ask_rects = [Rectangle((0, 0), 0, 0, color="#ff5555", alpha=0.7) for _ in range(10)]
        for r in self.book_bid_rects + self.book_ask_rects:
            ax.add_patch(r)
        self.book_title = ax.set_title("Book — empty", color="#cccccc", fontsize=10, loc="left")

    def draw_book(self, snap):
        ax = self.ax_book
        view = snap["market"]
        bids, asks = list(view.bids), list(view.asks)
        for r in self.book_bid_rects + self.book_ask_rects:
            r.set_visible(False)
        if not bids or not asks:
            self.book_title.set_text(f"Book {snap['focus']} — empty")
            ax.set_xlim(0, 1)
            return
        all_prices = [p for p, _ in bids + asks]
        price_span = max(all_prices) - min(all_prices)
        h = price_span / 28.0 if price_span > 0 else max(abs(all_prices[0]) * 0.0005, 1e-6)
        max_q = max([q for _, q in bids + asks] or [1.0])
        for rect, (p, q) in zip(self.book_bid_rects, bids):
            rect.set_xy((0, p - h / 2))
            rect.set_width(q)
            rect.set_height(h)
            rect.set_visible(True)
        for rect, (p, q) in zip(self.book_ask_rects, asks):
            rect.set_xy((0, p - h / 2))
            rect.set_width(q)
            rect.set_height(h)
            rect.set_visible(True)
        title = f"Book {snap['focus']}"
        if view.spread_bps is not None:
            title += f"  sp {view.spread_bps:.1f}bps"
        if view.book_imbalance is not None:
            title += f"  imb {view.book_imbalance:.2f}"
        self.book_title.set_text(title)
        ax.set_xlim(0, max_q * 1.15)
        ax.set_ylim(*_pad_limits(all_prices, frac=0.03))

    def _setup_tape(self):
        ax = self.ax_tape
        ax.set_xlabel("seconds ago", fontsize=8, color="#999")
        self._style_axes(ax)
        self.tape_zero = ax.axhline(0, color="#666", linewidth=0.6)
        self.tape_buy_fill = PolyCollection([_EMPTY_POLY], facecolors="#48d597", alpha=0.55, edgecolors="none")
        self.tape_sell_fill = PolyCollection([_EMPTY_POLY], facecolors="#ff5555", alpha=0.55, edgecolors="none")
        ax.add_collection(self.tape_buy_fill)
        ax.add_collection(self.tape_sell_fill)
        self.tape_title = ax.set_title("Tape pressure — no data", color="#cccccc", fontsize=10, loc="left")

    def draw_tape(self, snap):
        ax = self.ax_tape
        data = list(snap["market"].tape_series)
        if not data:
            _set_poly_between(self.tape_buy_fill, [], 0, [])
            _set_poly_between(self.tape_sell_fill, [], 0, [])
            self.tape_title.set_text(f"Tape pressure {snap['focus']} — no data")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            return
        n = snap["market"].ts
        xs = [(t - n) for t, _, _ in data]
        buys = [b for _, b, _ in data]
        sells = [-s for _, _, s in data]
        _set_poly_between(self.tape_buy_fill, xs, 0, buys)
        _set_poly_between(self.tape_sell_fill, xs, 0, sells)
        self.tape_title.set_text(f"Tape pressure {snap['focus']} (rolling {self.cfg.tape_window_sec:.0f}s window, $)")
        ax.set_xlim(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
        ax.set_ylim(*_pad_limits(buys + sells + [0.0]))

    def _setup_flow(self):
        ax = self.ax_flow
        self._style_axes(ax, grid=True, alpha=0.12)
        ax.set_xlabel("seconds ago", fontsize=8, color="#999")
        ax.set_ylabel("churn, turb×1e4", fontsize=8, color="#bbbbbb")
        ax.tick_params(axis="y", labelcolor="#dddddd")
        self.flow_line_churn, = ax.plot([], [], color="#f5b942", label="churn (impulse/depth)", linewidth=1.35)
        self.flow_line_turb, = ax.plot([], [], color="#c678ff", label="turb×1e4", linewidth=1.05, alpha=0.88)
        ax2 = ax.twinx()
        self.ax_flow_twin = ax2
        ax2.set_ylabel("log10(1+viscosity)", fontsize=8, color="#6eb5ff")
        ax2.tick_params(axis="y", labelcolor="#6eb5ff")
        for sp in ("top",):
            ax2.spines[sp].set_visible(False)
        self.flow_line_visc, = ax2.plot([], [], color="#6eb5ff", label="log10(1+visc)", linewidth=1.15)
        handles = [self.flow_line_churn, self.flow_line_turb, self.flow_line_visc]
        labs = [h.get_label() for h in handles]
        ax.legend(handles, labs, loc="upper left", fontsize=7, framealpha=0.4, labelcolor="#cccccc")
        self.flow_title = ax.set_title("Flow field — waiting", color="#cccccc", fontsize=10, loc="left")
        _set_axis_plain_numbers(ax)

    def draw_flow(self, snap):
        view = snap["market"]
        data = list(view.flow_series)
        sym = snap["focus"]
        n = view.ts
        ax = self.ax_flow
        if len(data) < 2:
            self.flow_line_churn.set_data([], [])
            self.flow_line_turb.set_data([], [])
            self.flow_line_visc.set_data([], [])
            self.flow_title.set_text(f"Flow field — {sym} (collecting…)")
            ax.set_xlim(-60, 1)
            ax.set_ylim(0, 1)
            self.ax_flow_twin.set_ylim(0, 1)
            return
        xs = [row[0] - n for row in data]
        churn = [row[1] for row in data]
        turb = [row[3] * 1e4 for row in data]
        visc_log = [math.log10(1.0 + max(row[2], 0.0)) for row in data]
        self.flow_line_churn.set_data(xs, churn)
        self.flow_line_turb.set_data(xs, turb)
        self.flow_line_visc.set_data(xs, visc_log)
        last_a = data[-1][4]
        last_s = data[-1][5]
        self.flow_title.set_text(
            f"Flow field — {sym}  last: churn={churn[-1]:.4f}  turb×1e4={turb[-1]:.4f}  "
            f"log10(1+visc)={visc_log[-1]:.2f}  accel={last_a:+.2f}  signed $/s={last_s:+.0f}"
        )
        ax.set_xlim(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
        ax.set_ylim(*_pad_limits(churn + turb, fallback=(0.0, 0.01)))
        self.ax_flow_twin.set_ylim(*_pad_limits(visc_log, fallback=(0.0, 0.5)))

    def _setup_performance(self):
        ax = self.ax_performance
        self._style_axes(ax, grid=True, alpha=0.12)
        ax.set_xlabel("minutes ago", fontsize=8, color="#999")
        ax.set_ylabel("net return / drawdown %", fontsize=8, color="#bbbbbb")
        ax.tick_params(axis="y", labelcolor="#dddddd")
        self.perf_net_line, = ax.plot([], [], color="#48d597", label="net return %", linewidth=1.35)
        self.perf_dd_line, = ax.plot([], [], color="#ff5555", label="drawdown %", linewidth=1.05, alpha=0.86)
        ax2 = ax.twinx()
        self.ax_perf_twin = ax2
        ax2.set_ylabel("return velocity %/h", fontsize=8, color="#6eb5ff")
        ax2.tick_params(axis="y", labelcolor="#6eb5ff")
        for sp in ("top",):
            ax2.spines[sp].set_visible(False)
        self.perf_rph_line, = ax2.plot([], [], color="#6eb5ff", label="session %/h", linewidth=1.15)
        self.perf_eph_line, = ax2.plot([], [], color="#f5b942", label="exposure %/h", linewidth=1.15, alpha=0.9)
        handles = [self.perf_net_line, self.perf_dd_line, self.perf_rph_line, self.perf_eph_line]
        labs = [h.get_label() for h in handles]
        ax.legend(handles, labs, loc="upper left", fontsize=7, framealpha=0.4, labelcolor="#cccccc")
        self.perf_title = ax.set_title("Return velocity — waiting", color="#cccccc", fontsize=10, loc="left")
        _set_axis_plain_numbers(ax)
        _set_axis_plain_numbers(ax2)

    def draw_performance(self, snap):
        ax = self.ax_performance
        data = list(snap["market"].performance_history)
        if len(data) < 2:
            for line in (self.perf_net_line, self.perf_dd_line, self.perf_rph_line, self.perf_eph_line):
                line.set_data([], [])
            self.perf_title.set_text("Return velocity — collecting…")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            self.ax_perf_twin.set_ylim(-1, 1)
            return
        n = snap["market"].ts
        xs = [(row[0] - n) / 60.0 for row in data]
        net = [row[1] * 100.0 for row in data]
        session_v = [row[2] * 100.0 for row in data]
        exposure_v = [row[3] * 100.0 for row in data]
        drawdown = [-row[4] * 100.0 for row in data]
        self.perf_net_line.set_data(xs, net)
        self.perf_dd_line.set_data(xs, drawdown)
        self.perf_rph_line.set_data(xs, session_v)
        self.perf_eph_line.set_data(xs, exposure_v)
        last = data[-1]
        avg_hold_min = last[6] / 60.0
        self.perf_title.set_text(
            f"Return velocity  net={net[-1]:+.2f}%  session={session_v[-1]:+.2f}%/h  "
            f"exposure={exposure_v[-1]:+.2f}%/h  dd={-drawdown[-1]:.2f}%  "
            f"hold_avg={avg_hold_min:.1f}m  trade_vel={last[8]:+.3f}%/m  "
            f"horizon_score={last[5]:+.2f}"
        )
        ax.set_xlim(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
        ax.set_ylim(*_pad_limits(net + drawdown + [0.0], fallback=(-1.0, 1.0)))
        self.ax_perf_twin.set_ylim(*_pad_limits(session_v + exposure_v + [0.0], fallback=(-1.0, 1.0)))

    def _setup_gauge(self, ax, suffix, labels, fmts, regime_name):
        y = list(range(len(labels)))
        bars = ax.barh(y, [0.0] * len(labels), color=["#444"] * len(labels), edgecolor="#222", height=0.7)
        ax.axvline(1.0, color="#ffffff", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 3.4)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["0", "thr", "2x"], fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        title = ax.set_title("", color="#cccccc", fontsize=10, loc="left")
        val_texts = [ax.text(2.15, i, "", va="center", ha="left", fontsize=8, color="#ffffff", weight="bold") for i in range(len(labels))]
        thr_texts = [ax.text(3.35, i, "", va="center", ha="right", fontsize=7.5, color="#888888") for i in range(len(labels))]
        setattr(self, f"bars{suffix}", bars)
        setattr(self, f"vtxt{suffix}", val_texts)
        setattr(self, f"ttxt{suffix}", thr_texts)
        setattr(self, f"title{suffix}", title)
        setattr(self, f"fmts{suffix}", fmts)
        setattr(self, f"regime{suffix}", regime_name)

    def _draw_gauge(self, suffix, sym, values, thresholds, greater):
        bars = getattr(self, f"bars{suffix}")
        vtxts = getattr(self, f"vtxt{suffix}")
        ttxts = getattr(self, f"ttxt{suffix}")
        title = getattr(self, f"title{suffix}")
        fmts = getattr(self, f"fmts{suffix}")
        regime_name = getattr(self, f"regime{suffix}")
        for i, (v, t, bar, vt, tt) in enumerate(zip(values, thresholds, bars, vtxts, ttxts)):
            fmt = fmts[i] if i < len(fmts) else "{:.2f}"
            if v is None or t in (None, 0):
                bar.set_width(0)
                bar.set_color("#444")
                vt.set_text("—")
                tt.set_text("")
                continue
            ratio = v / t if t else 0
            bar.set_width(min(max(ratio, 0.0), 2.0))
            ok = (v >= t) if greater[i] else (v <= t)
            bar.set_color("#48d597" if ok else "#ff5555")
            vt.set_text(fmt.format(v))
            tt.set_text(f"thr {fmt.format(t)}")
        title.set_text(f"{regime_name} — {sym}   (bar: value÷threshold)")

    def draw_micro_gauges(self, snap):
        s = snap["market"].micro or {}
        max_spread = s.get("max_spread", 12.0)
        values = [
            s.get("trades"),
            s.get("buy_not"),
            s.get("imbalance"),
            s.get("burst_ratio"),
            (s.get("move_pct") or 0) * 100.0 if s else None,
            s.get("book_imb"),
            s.get("spread_bps"),
        ]
        thresholds = [
            self.cfg.micro_min_trades,
            self.cfg.micro_min_buy_notional,
            self.cfg.micro_min_imbalance,
            self.cfg.micro_burst_multiple,
            self.cfg.micro_min_move_pct * 100.0,
            1.08,
            max_spread,
        ]
        self._draw_gauge("_micro", snap["focus"], values, thresholds, MICRO_GREATER)

    def draw_thrust_gauges(self, snap):
        s = snap["market"].micro or {}
        values = [
            (s.get("thrust_r3") or 0) * 100.0 if s else None,
            (s.get("thrust_r5") or 0) * 100.0 if s else None,
            (s.get("thrust_r15") or 0) * 100.0 if s else None,
            s.get("thrust_vol_x"),
            s.get("thrust_close_pos"),
        ]
        thresholds = [
            self.cfg.thrust_min_ret_3m * 100.0,
            self.cfg.thrust_min_ret_5m * 100.0,
            0.000001,
            self.cfg.thrust_min_vol_x,
            self.cfg.thrust_min_close_pos,
        ]
        self._draw_gauge("_thrust", snap["focus"], values, thresholds, THRUST_GREATER)

    def _setup_text_panel(self, ax, suffix, title, mono=False, fontsize=9):
        ax.axis("off")
        title_artist = ax.set_title(title, color="#cccccc", fontsize=10, loc="left")
        family = "monospace" if mono else "DejaVu Sans"
        text_artist = ax.text(0.01, 0.97, "", transform=ax.transAxes, fontsize=fontsize, va="top", ha="left", color="#dddddd", family=family, clip_on=True)
        setattr(self, f"title{suffix}", title_artist)
        setattr(self, f"text{suffix}", text_artist)

    def draw_status(self, snap):
        lines = [f"MODE: {snap['mode']}"]
        lines.append(f"BTC context: {'OK' if snap['btc_ok'] else 'WEAK'} | strict {'OK' if snap['btc_strict'] else 'NO'} | flush {'YES' if snap['btc_flush'] else 'no'}")
        perf = snap.get("performance") or {}
        if perf:
            lines.append(
                f"RETURN VELOCITY: net {perf.get('net_return', 0.0)*100:+.2f}% | "
                f"session {perf.get('return_per_hour', 0.0)*100:+.2f}%/h | "
                f"exposure {perf.get('return_per_exposure_hour', 0.0)*100:+.2f}%/h | "
                f"avg hold {perf.get('avg_hold_sec', 0.0)/60.0:.1f}m | "
                f"trade vel {perf.get('avg_return_velocity_pct_per_min', 0.0):+.3f}%/m | "
                f"score {perf.get('horizon_score', 0.0):+.2f}"
            )
        pos = snap.get("position")
        if pos:
            mark = snap["market"].last_price or pos["entry_signal"]
            # Display-only approximation; trader's true net uses L2 fill estimate.
            net = mark / max(pos["entry"], 1e-9) - 1.0
            r = self.cfg.risk(pos["regime"])
            age = int(snap["market"].ts - pos["entry_time"])
            lines.append(f"POSITION  {pos['pair']}  [{pos['regime']}]  approx {net*100:+.2f}%  trail {'ON' if pos['trail_active'] else 'off'}  age {age}s  hold≤{r['max_hold']:.0f}s")
            lines.append(f"  entry {pos['entry']:.6g}  peak {pos['peak_mark']:.6g}  qty {pos['qty']:.6g}")
            lines.append(f"  stop -{r['hard_stop']*100:.1f}% / activate +{r['trail_activate']*100:.1f}% / trail {r['trail_pct']*100:.1f}% / TP {r['take_profit']*100:.1f}%")
            lines.append(f"  note: {pos['note']}")
        else:
            lines.append(f"FLAT  cash ${snap['cash']:.4f}")
        heat = 0.0
        if pos and snap["equity"] > 0:
            heat = pos["qty"] * (snap["market"].last_price or pos["entry_signal"]) / snap["equity"]
        lines.append(f"HEAT: {heat:.1%} / {self.cfg.max_portfolio_heat:.1%}")

        if snap["watch"]:
            n = snap["market"].ts
            ws = []
            for s, w in list(snap["watch"])[:6]:
                ws.append(f"{s} +{w['spike_pct']:.1f}%/x{w['burst_x']:.1f} ({int(n-w['at'])}s)")
            lines.append("Watch: " + " | ".join(ws))
        if snap["candidates"]:
            cs = sorted(snap["candidates"], key=lambda c: -c["score"])[:4]
            lines.append("Candidates: " + " | ".join([f"{c['symbol']} [{c['regime']}] score {c['score']:.2f}" for c in cs]))
        auto = snap.get("auto") or {}
        if auto:
            lines.append(f"AUTO_OPT stats: dd={auto.get('dd_pct',0):.2f}% idle={auto.get('idle_min',0):.1f}m missed={auto.get('missed',0)} rej={auto.get('rejected',0)} exp={auto.get('expired',0)}")
        lines.append("")
        lines.append("keys: o=optuna  r=recordings  1/2/3=focus  ?=help")
        self.text_status.set_text("\n".join(lines))

    def draw_trades_or_optuna(self, snap):
        opt = snap.get("optuna") or {}
        show_optuna = opt.get("running") or opt.get("completed", 0) > 0 or opt.get("stage") in ("error", "done", "validating", "splitting")
        if show_optuna:
            self._draw_optuna_panel(opt)
            return
        self.title_trades.set_text("Recent trades")
        self.title_trades.set_color("#cccccc")
        trades = list(snap["trades"])
        if not trades:
            self.text_trades.set_text("no trades yet  —  press 'o' to start an Optuna study")
            self.text_trades.set_color("#888")
            return
        recent = trades[-6:][::-1]
        header = f"{'time':<9}{'pair':<10}{'regime':<12}{'reason':<20}{'pnl':>9}{'ret%':>8}{'hold':>7}{'vel%/m':>9}"
        lines = [header]
        for t in recent:
            etime = str(t.get("exit_time", ""))[11:19]
            regime = str(t.get("regime", ""))[:11]
            hold_min = float(t.get("hold_sec", 0.0) or 0.0) / 60.0
            lines.append(
                f"{etime:<9}{t['pair']:<10}{regime:<12}{str(t.get('reason',''))[:19]:<20}"
                f"{t['pnl_usd']:>+9.4f}{float(t.get('return_pct', 0.0) or 0.0):>+8.2f}"
                f"{hold_min:>6.1f}m{float(t.get('return_velocity_pct_per_min', 0.0) or 0.0):>+9.3f}"
            )
        self.text_trades.set_text("\n".join(lines))
        self.text_trades.set_color("#dddddd")

    def _draw_optuna_panel(self, opt):
        running = bool(opt.get("running"))
        status = "RUNNING" if running else str(opt.get("stage", "DONE")).upper()
        if opt.get("stage") == "error":
            color = "#ff5555"
        elif running:
            color = "#48d597"
        else:
            color = "#f5b942"
        done = opt.get("completed", 0)
        total = opt.get("total", 0)
        header = f"Optuna study  [{status}]  {self._progress_bar(done, total, 22)}  {done}/{total or '?'}"
        self.title_trades.set_text(header)
        self.title_trades.set_color(color)
        elapsed = opt.get("elapsed", 0.0) or 0.0
        lines = []
        if opt.get("replay_path"):
            lines.append(f"replay: {opt['replay_path']}   elapsed {elapsed:.1f}s")
        lines.append(f"stage: {opt.get('stage','')}   {opt.get('message','')}")
        if opt.get("last_error"):
            lines.append(f"last error: {opt['last_error']}")
        if opt.get("current_trial") is not None:
            lines.append(f"current: trial {opt.get('current_trial')} file {opt.get('current_file','')}")
        if opt.get("trigger_note"):
            lines.append(f"trigger: {opt['trigger_note']}")
        if opt.get("best_score") is not None:
            lines.append(f"BEST  trial {opt.get('best_trial'):>3}  score {opt.get('best_score'):+.4f}")
        else:
            lines.append("BEST  (no trials yet)")
        if opt.get("baseline_validation_aggregate") is not None:
            lines.append(f"validation: candidate {opt.get('candidate_validation_aggregate'):+.3f} vs baseline {opt.get('baseline_validation_aggregate'):+.3f} promoted={opt.get('promoted')}")
        lines.append("")
        n_recordings = len(opt.get("replay_paths") or [])
        if n_recordings > 1:
            lines.append(f"{'#':>4} {'score':>9} {'min':>8} {'tr':>3} {'net%':>7} {'r/h':>7} {'e/h':>7} {'hold':>6} {'dd%':>5}")
        else:
            lines.append(f"{'#':>4} {'score':>9} {'Δbest':>9} {'tr':>3} {'net%':>7} {'r/h':>7} {'e/h':>7} {'hold':>6} {'dd%':>5}")
        best = opt.get("best_score")
        for r in (opt.get("trials") or [])[-7:]:
            a = r.get("attrs", {})
            score = r.get("score") or 0.0
            mark = "★" if r.get("is_best") else " "
            hold_min = float(a.get('avg_hold_sec', 0.0) or 0.0) / 60.0
            if n_recordings > 1:
                lines.append(
                    f"{mark}{r['n']:>3} {score:>+9.4f} {a.get('min_score',0):>+8.3f} "
                    f"{a.get('trades',0):>3} {a.get('net_return',0)*100:>+7.2f} "
                    f"{a.get('return_per_hour',0)*100:>+7.2f} {a.get('return_per_exposure_hour',0)*100:>+7.2f} "
                    f"{hold_min:>5.1f}m {a.get('max_drawdown',0)*100:>5.2f}"
                )
            else:
                delta = (score - best) if best is not None else 0.0
                lines.append(
                    f"{mark}{r['n']:>3} {score:>+9.4f} {delta:>+9.4f} "
                    f"{a.get('trades',0):>3} {a.get('net_return',0)*100:>+7.2f} "
                    f"{a.get('return_per_hour',0)*100:>+7.2f} {a.get('return_per_exposure_hour',0)*100:>+7.2f} "
                    f"{hold_min:>5.1f}m {a.get('max_drawdown',0)*100:>5.2f}"
                )
        self.text_trades.set_text("\n".join(lines))
        self.text_trades.set_color("#dddddd")

    @staticmethod
    def _progress_bar(done, total, width=20):
        if not total:
            return "[" + "·" * width + "]"
        filled = int(round(width * done / total))
        return "[" + "█" * filled + "·" * (width - filled) + "]"

    def draw_regime_panel(self, snap):
        s = snap["market"].micro or {}
        be_pct = breakeven_move_pct(self.cfg) * 100.0
        baseline_buy = (s.get("buy_not") or 0) / max(s.get("burst_ratio") or 1, 1e-9)
        flow_keys = {k: v for k, v in s.items() if str(k).startswith("flow_")}
        info = [
            f"focus: {snap['focus']}",
            "  [1/2/3=lock 0=auto]",
            f"breakeven fallback: {be_pct:.2f}%",
            f"tape n: {s.get('trades', 0)}",
            f"buy 12s: ${s.get('buy_not', 0):,.0f}",
            f"base/12s: ${baseline_buy:,.0f}",
            f"params: {snap['params_source']}",
            "",
            *format_lines(flow_keys),
        ]
        self.text_regime.set_text("\n".join(info))

    def update(self, _frame):
        opt = self.optuna.snapshot() if self.optuna is not None else {}
        snap = self.engine.dashboard_snapshot(focus=self.locked_focus, mode="LIVE", optuna=opt, params_source=self.params_source)
        self.focus = snap["focus"]
        try:
            self.draw_price(snap)
            self.draw_book(snap)
            self.draw_tape(snap)
            self.draw_flow(snap)
            self.draw_performance(snap)
            self.draw_micro_gauges(snap)
            self.draw_thrust_gauges(snap)
            self.draw_capital(snap)
            self.draw_status(snap)
            self.draw_trades_or_optuna(snap)
            self.draw_regime_panel(snap)
        except Exception as e:
            print(f"  [DASH] draw error: {e}")
        return []

    def show(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=150, cache_frame_data=False)
        plt.show()



