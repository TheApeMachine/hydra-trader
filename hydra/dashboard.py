from __future__ import annotations

import math
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from matplotlib.ticker import ScalarFormatter

from .constants import BTC_SYMBOL, FORCE_SYMBOLS

# Verdict palette — used everywhere so the dashboard speaks one color language.
COLOR_POS = "#48d597"   # green: passing, making money, prediction tracking
COLOR_NEU = "#bbbbbb"   # grey: waiting, no signal
COLOR_NEG = "#ff5555"   # red: blocked, losing
COLOR_DIM = "#666666"
COLOR_TEXT = "#dddddd"
COLOR_SOFT = "#999999"

# Stable per-symbol colours for the multi-symbol chart and gauge needles.
SYMBOL_COLORS = {
    "BTC/USD": "#f5b942",   # gold
    "ETH/USD": "#6eb5ff",   # blue
    "SOL/USD": "#c678ff",   # purple
}
EXTRA_PALETTE = ["#48d597", "#ff7e7e", "#7fe1c8", "#ffb88c", "#a3a3ff", "#ffd166"]

EMPTY_OFFSETS = np.empty((0, 2))
EMPTY_POLY = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

# Map raw trader reason codes to plain-English fragments.
REASON_PHRASES = {
    "take_profit": "hit take-profit",
    "hard_stop_net": "loss guard tripped",
    "hard_stop": "loss guard tripped",
    "trail": "trailing stop locked profit",
    "trail_loss": "trailing stop closed it",
    "timeout": "held too long, time was up",
    "macro_early_fail": "setup never confirmed",
    "btc_flush": "BTC dumped, exited for safety",
    "hawkes_sell_flip": "crowd flipped to selling",
    "velocity_stall": "stopped going up",
    "manual": "closed by you",
    "eob": "end of session",
    "spread_blowout": "spread got too expensive",
}

REGIME_PHRASES = {
    "book_ignition": "order-book ignition",
    "macro_reclaim_v2": "pullback reclaim",
    "macro_thrust": "momentum thrust",
}

# Short visible badge per Hawkes-learning status. Keep the surface tiny so the
# legend stays readable.
_LEARNING_BADGES = {
    "cold": "cold",
    "learning": "learning",
    "calibrated": "✓ CALIBRATED",
    "degraded": "↓ degraded",
    "noisy": "noisy",
    "inverted": "⚠ INVERTED",
}


def _set_axis_plain(ax):
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


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


def _verdict_color(verdict: str) -> str:
    if verdict == "positive":
        return COLOR_POS
    if verdict == "negative":
        return COLOR_NEG
    return COLOR_NEU


def _symbol_color(sym: str, idx: int) -> str:
    if sym in SYMBOL_COLORS:
        return SYMBOL_COLORS[sym]
    return EXTRA_PALETTE[idx % len(EXTRA_PALETTE)]


def _plain_reason(code: str) -> str:
    if not code:
        return "closed"
    base = str(code).strip()
    for key, phrase in REASON_PHRASES.items():
        if base.startswith(key):
            return phrase
    return base.replace("_", " ")


def _plain_regime(code: str) -> str:
    return REGIME_PHRASES.get(str(code or "").strip(), str(code or "").replace("_", " "))


# ── Single source of truth for the readiness score / gauge label ─────────
# Both the needle position AND the state label MUST use these functions, so
# they cannot disagree.

# Label thresholds (boundaries are inclusive on the upper bound).
_BUY_NOW_THRESHOLD = 60.0
_ALMOST_THRESHOLD = 20.0
_STAY_OUT_THRESHOLD = -40.0


def _label_for_score(score: float) -> tuple[str, str]:
    """Map a readiness score in [-100, +100] → (label, verdict)."""
    if score >= _BUY_NOW_THRESHOLD:
        return ("BUYING SOON", "positive")
    if score >= _ALMOST_THRESHOLD:
        return ("ALMOST", "neutral")
    if score <= _STAY_OUT_THRESHOLD:
        return ("STAY OUT", "negative")
    return ("WAITING", "neutral")


# Single threshold for "flat" in the market-overview panel. Both the title
# verdict and the per-symbol caption tags use this — they cannot disagree.
_MARKET_FLAT_THRESHOLD_PCT = 0.1


def _per_symbol_change_label(change_pct: float) -> str:
    """Return one of {'up', 'flat', 'down'} for a given % change."""
    if change_pct > _MARKET_FLAT_THRESHOLD_PCT:
        return "up"
    if change_pct < -_MARKET_FLAT_THRESHOLD_PCT:
        return "down"
    return "flat"


def _market_overview_verdict(per_symbol_change: dict[str, float]) -> tuple[str, str]:
    """Return (mood, verdict) from per-symbol % changes using the SAME flat
    threshold as the caption tags. 'broad' requires ≥2 symbols and a majority.
    """
    ups = sum(1 for v in per_symbol_change.values()
              if _per_symbol_change_label(v) == "up")
    downs = sum(1 for v in per_symbol_change.values()
                if _per_symbol_change_label(v) == "down")
    if ups >= 2 and ups > downs:
        return ("broad strength", "positive")
    if downs >= 2 and downs > ups:
        return ("broad weakness", "negative")
    return ("mixed", "neutral")


def _hawkes_hit_aggregate(per_symbol: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Sum per-symbol (agree, total) tuples into a single (agree_sum, total_sum).

    The title verdict MUST use this aggregate so it equals the sum visible in
    the per-symbol legend badges. Otherwise the panel contradicts itself.
    """
    agree = sum(int(a) for a, _ in per_symbol.values())
    total = sum(int(t) for _, t in per_symbol.values())
    return (agree, total)


from .readiness import symbol_readiness_score as _symbol_readiness_score


class Dashboard:
    """Glasses-friendly redesigned dashboard.

    Six panels, each one shows positive / neutral / negative at a glance via a
    coloured dot beside its title, plus a short plain-English explanation.
    Caller-facing surface (constructor signature, ``show()``) is unchanged so
    cli.py keeps working.
    """

    HELP_LINES = [
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
        # Multi-symbol chart removes the need for a locked focus, but keep the
        # attribute so engine snapshot calls stay quiet.
        self.locked_focus: str | None = None
        self.focus = BTC_SYMBOL
        # Remember pump events we've already labelled, so annotation text
        # placement doesn't churn every frame.
        self._annot_cache: dict[tuple[str, float], str] = {}

        self.fig = plt.figure(figsize=(17, 11))
        self.fig.canvas.manager.set_window_title(
            f"Hydra Trader — LIVE [{params_source}]{title_suffix}"
        )
        # 3 rows × 2 cols. Top row tallest (chart + gauge), middle medium
        # (hawkes + strategies), bottom shortest (status + trades).
        gs = GridSpec(
            3, 2, figure=self.fig,
            height_ratios=[1.6, 1.2, 1.1],
            width_ratios=[1.4, 1.0],
            left=0.055, right=0.985, top=0.945, bottom=0.045,
            hspace=0.42, wspace=0.18,
        )
        self.ax_chart = self.fig.add_subplot(gs[0, 0])
        self.ax_gauge = self.fig.add_subplot(gs[0, 1])
        self.ax_hawkes = self.fig.add_subplot(gs[1, 0])
        self.ax_strats = self.fig.add_subplot(gs[1, 1])
        self.ax_status = self.fig.add_subplot(gs[2, 0])
        self.ax_trades = self.fig.add_subplot(gs[2, 1])

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._setup_chart()
        self._setup_gauge()
        self._setup_hawkes()
        self._setup_strats()
        self._setup_text_panel(self.ax_status, "_status", "What Hydra is doing", fontsize=9.5)
        self._setup_text_panel(self.ax_trades, "_trades", "Recent trades", fontsize=9.0, mono=False)

    # ── keys ────────────────────────────────────────────────
    def on_key(self, event):
        k = event.key
        if k == "o":
            self.toggle_optuna()
            return
        if k == "r":
            from .data import list_recordings
            import os as _os
            recs = list_recordings()
            if not recs:
                print("[DASH] no recordings found in ./runs/ or .")
            else:
                print("[DASH] recordings (newest first):")
                for p in recs:
                    age_min = (time.time() - _os.path.getmtime(p)) / 60.0
                    sz = _os.path.getsize(p) / 1024.0
                    print(f"   {p}  ({sz:.1f} KB, {age_min:.1f} min ago)")
            return
        if k == "?":
            print("\n[DASH] keybindings:")
            for line in self.HELP_LINES:
                print("   " + line)
            return

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

    # ── shared helpers ──────────────────────────────────────
    @staticmethod
    def _style_axes(ax, grid=True, alpha=0.15):
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(axis="both", labelsize=8, colors=COLOR_SOFT)
        if grid:
            ax.grid(True, alpha=alpha)
        _set_axis_plain(ax)

    def _set_panel_title(self, title_artist, dot_artist, text: str, verdict: str):
        title_artist.set_text("   " + text)  # space for the dot
        col = _verdict_color(verdict)
        dot_artist.set_color(col)
        dot_artist.set_visible(True)

    @staticmethod
    def _set_offsets_safe(scatter, points):
        scatter.set_offsets(np.asarray(points, dtype=float) if points else EMPTY_OFFSETS)

    # ── [1] multi-symbol chart ──────────────────────────────
    def _setup_chart(self):
        ax = self.ax_chart
        self._style_axes(ax)
        ax.set_xlabel("minutes ago", fontsize=8, color=COLOR_SOFT)
        ax.set_ylabel("% from start of view", fontsize=8, color=COLOR_SOFT)
        ax.axhline(0, color="#444", linewidth=0.6, alpha=0.7)
        # Up to 8 reusable line artists; we re-label them each frame.
        self.chart_lines: dict[str, Any] = {}
        self._chart_pool: list[Any] = []
        for i in range(8):
            ln, = ax.plot([], [], linewidth=1.4, color=EXTRA_PALETTE[i % len(EXTRA_PALETTE)])
            ln.set_visible(False)
            self._chart_pool.append(ln)
        self.chart_legend = None
        # Annotation pool — drawn as scatter markers + text artists.
        self.chart_annot_scatter = ax.scatter([], [], marker="^", s=70, color=COLOR_POS,
                                              edgecolors="white", linewidths=0.6, zorder=5)
        self._chart_annot_texts: list[Any] = []
        for _ in range(8):
            t = ax.text(0, 0, "", fontsize=7.5, color=COLOR_TEXT, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a1a1a",
                                  edgecolor="#444", alpha=0.85))
            t.set_visible(False)
            self._chart_annot_texts.append(t)
        # Title + verdict dot
        self.title_chart = ax.set_title("", color=COLOR_TEXT, fontsize=11, loc="left")
        self.dot_chart = ax.text(0.005, 1.025, "●", transform=ax.transAxes,
                                 fontsize=14, color=COLOR_NEU, va="bottom", ha="left")
        # Caption beneath the chart with one-line summary
        self.caption_chart = ax.text(
            0.005, -0.18, "", transform=ax.transAxes, fontsize=8.5, color=COLOR_SOFT,
            va="top", ha="left",
        )

    def draw_chart(self, snap):
        ax = self.ax_chart
        multi = snap["market"].multi_prices or {}
        n = snap["market"].ts

        # Compute per-symbol % return from earliest sample in this view.
        any_data = False
        all_xs: list[float] = []
        all_ys: list[float] = []
        per_symbol_change: dict[str, float] = {}

        # Reset pool
        for ln in self._chart_pool:
            ln.set_visible(False)
            ln.set_data([], [])

        # Build a stable display order: forced first, then watchlist symbols.
        ordered = [s for s in FORCE_SYMBOLS if s in multi]
        for s in multi.keys():
            if s not in ordered:
                ordered.append(s)
        ordered = ordered[:len(self._chart_pool)]

        active_handles = []
        active_labels = []
        for i, sym in enumerate(ordered):
            series = multi.get(sym) or ()
            if len(series) < 2:
                continue
            ln = self._chart_pool[i]
            color = _symbol_color(sym, i)
            xs = [(t - n) / 60.0 for t, _ in series]
            base = series[0][1]
            if base <= 0:
                continue
            ys = [(p / base - 1.0) * 100.0 for _, p in series]
            ln.set_data(xs, ys)
            ln.set_color(color)
            ln.set_visible(True)
            any_data = True
            all_xs.extend(xs)
            all_ys.extend(ys)
            per_symbol_change[sym] = ys[-1]
            active_handles.append(ln)
            short = sym.split("/")[0]
            active_labels.append(f"{short} {ys[-1]:+.2f}%")

        # Legend refresh
        if self.chart_legend is not None:
            try:
                self.chart_legend.remove()
            except Exception:
                pass
            self.chart_legend = None
        if active_handles:
            self.chart_legend = ax.legend(
                active_handles, active_labels, loc="upper left",
                fontsize=8, framealpha=0.45, labelcolor="#dddddd",
                ncol=min(len(active_handles), 4),
            )

        if not any_data:
            self._set_panel_title(self.title_chart, self.dot_chart,
                                  "Market overview — waiting for ticks", "neutral")
            self.caption_chart.set_text("")
            self.chart_annot_scatter.set_offsets(EMPTY_OFFSETS)
            for t in self._chart_annot_texts:
                t.set_visible(False)
            ax.set_xlim(-5, 0)
            ax.set_ylim(-1, 1)
            return

        # Axes limits
        xlo, xhi = min(all_xs), max(all_xs)
        if xhi <= xlo:
            xhi = xlo + 1
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(*_pad_limits(all_ys + [0.0], frac=0.12))

        # Annotations from pump events (cap to 6 most recent in-window)
        events = list(snap.get("pump_events") or [])
        events.sort(key=lambda e: -float(e.get("at", 0.0)))
        annot_points = []
        annot_iter = iter(self._chart_annot_texts)
        used_texts: list[Any] = []
        for ev in events:
            sym = ev.get("symbol")
            if sym not in multi:
                continue
            series = multi[sym]
            ev_t = float(ev.get("at", 0.0))
            x = (ev_t - n) / 60.0
            if x < xlo or x > xhi:
                continue
            # Find the nearest price point to place the annotation marker
            nearest = min(series, key=lambda row: abs(row[0] - ev_t))
            base = series[0][1]
            if base <= 0:
                continue
            y = (nearest[1] / base - 1.0) * 100.0
            annot_points.append((x, y))
            label = f"⚡{sym.split('/')[0]} +{float(ev.get('spike_pct', 0.0)):.1f}%"
            try:
                txt = next(annot_iter)
            except StopIteration:
                break
            txt.set_position((x, y))
            txt.set_text(label)
            txt.set_visible(True)
            used_texts.append(txt)
            if len(used_texts) >= 6:
                break
        # Hide unused annotation slots
        for t in self._chart_annot_texts:
            if t not in used_texts:
                t.set_visible(False)
        self.chart_annot_scatter.set_offsets(
            np.asarray(annot_points, dtype=float) if annot_points else EMPTY_OFFSETS
        )

        # Verdict and caption tags use the SAME flat threshold so they
        # cannot disagree.
        mood, verdict = _market_overview_verdict(per_symbol_change)
        self._set_panel_title(self.title_chart, self.dot_chart,
                              f"Market overview — {mood}", verdict)
        bits = [
            f"{sym.split('/')[0]} {_per_symbol_change_label(ch)}"
            for sym, ch in per_symbol_change.items()
        ]
        cap = "Each line is one coin, % change since the start of this view. "
        if bits:
            cap += "Right now: " + ", ".join(bits[:6]) + "."
        if events:
            cap += f"  ⚡ marks where a coin recently spiked."
        self.caption_chart.set_text(cap)

    # ── [2] multi-needle readiness gauge ────────────────────
    def _setup_gauge(self):
        ax = self.ax_gauge
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.4, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        # Gauge band: red (left) → grey (middle) → green (right)
        # Drawn as wedges 180° → 0° (left to right).
        red = Wedge((0, 0), 1.0, 150, 180, width=0.18, facecolor=COLOR_NEG, edgecolor="#1a1a1a", alpha=0.85)
        amber_neg = Wedge((0, 0), 1.0, 120, 150, width=0.18, facecolor="#ff9966", edgecolor="#1a1a1a", alpha=0.8)
        neu_neg = Wedge((0, 0), 1.0, 90, 120, width=0.18, facecolor="#666", edgecolor="#1a1a1a", alpha=0.6)
        neu_pos = Wedge((0, 0), 1.0, 60, 90, width=0.18, facecolor="#666", edgecolor="#1a1a1a", alpha=0.6)
        amber_pos = Wedge((0, 0), 1.0, 30, 60, width=0.18, facecolor="#cdd96a", edgecolor="#1a1a1a", alpha=0.8)
        green = Wedge((0, 0), 1.0, 0, 30, width=0.18, facecolor=COLOR_POS, edgecolor="#1a1a1a", alpha=0.85)
        for w in (red, amber_neg, neu_neg, neu_pos, amber_pos, green):
            ax.add_patch(w)
        # Anchor labels
        ax.text(-1.02, -0.05, "SELL", color=COLOR_NEG, fontsize=8.5, ha="center", va="top", weight="bold")
        ax.text(0.0, 1.05, "WAIT", color=COLOR_NEU, fontsize=8.5, ha="center", va="bottom")
        ax.text(1.02, -0.05, "BUY", color=COLOR_POS, fontsize=8.5, ha="center", va="top", weight="bold")
        # Needle pool (we draw lines from center to perimeter)
        self._needle_pool: list[Any] = []
        self._needle_labels: list[Any] = []
        for i in range(6):
            ln, = ax.plot([0, 0], [0, 0], linewidth=2.4, color=COLOR_NEU, solid_capstyle="round")
            ln.set_visible(False)
            self._needle_pool.append(ln)
            lab = ax.text(0, 0, "", fontsize=8, color=COLOR_TEXT, ha="center", va="center", weight="bold")
            lab.set_visible(False)
            self._needle_labels.append(lab)
        # State label (big text under the gauge)
        self.gauge_state = ax.text(0, -0.18, "", fontsize=14, color=COLOR_TEXT,
                                   ha="center", va="top", weight="bold")
        self.gauge_substate = ax.text(0, -0.32, "", fontsize=9, color=COLOR_SOFT,
                                      ha="center", va="top", wrap=True)
        # Title + dot
        self.title_gauge = ax.set_title("", color=COLOR_TEXT, fontsize=11, loc="left")
        self.dot_gauge = ax.text(0.005, 1.025, "●", transform=ax.transAxes,
                                 fontsize=14, color=COLOR_NEU, va="bottom", ha="left")

    def _score_to_angle(self, score: float) -> float:
        # Map [-100, +100] → [180°, 0°] (left to right)
        s = max(-100.0, min(100.0, score))
        return 180.0 - (s + 100.0) / 200.0 * 180.0

    def draw_gauge(self, snap):
        # Determine which symbols get needles: forced + watch + held.
        symbols: list[str] = []
        for s in FORCE_SYMBOLS:
            symbols.append(s)
        for s, _ in (snap.get("watch") or ()):
            if s not in symbols:
                symbols.append(s)
        pos = snap.get("position")
        if pos and pos["pair"] not in symbols:
            symbols.append(pos["pair"])
        symbols = symbols[:len(self._needle_pool)]

        # Hide all needles, then draw active ones.
        for ln in self._needle_pool:
            ln.set_visible(False)
        for lab in self._needle_labels:
            lab.set_visible(False)

        rendered: list[tuple[str, float]] = []
        for i, sym in enumerate(symbols):
            score = _symbol_readiness_score(sym, snap)
            angle_deg = self._score_to_angle(score)
            angle = math.radians(angle_deg)
            r = 0.78
            x, y = r * math.cos(angle), r * math.sin(angle)
            ln = self._needle_pool[i]
            color = _symbol_color(sym, i)
            ln.set_data([0, x], [0, y])
            ln.set_color(color)
            ln.set_visible(True)
            lab = self._needle_labels[i]
            label_r = 0.92
            lab.set_position((label_r * math.cos(angle), label_r * math.sin(angle)))
            lab.set_text(sym.split("/")[0])
            lab.set_color(color)
            lab.set_visible(True)
            rendered.append((sym, score))

        # State label MUST be derived from the same score as the needle so
        # the two cannot contradict each other. The leading symbol is the one
        # with the largest |score|. If a position is open, that symbol wins.
        state_text = "WAITING"
        sub_text = "No symbol close to a trade right now."
        verdict = "neutral"
        inertia_view = snap["market"].inertia_state or {}
        if pos:
            mark = snap["market"].last_price or pos["entry_signal"]
            net = mark / max(pos["entry"], 1e-9) - 1.0
            stop_px = pos.get("stop_price")
            if stop_px and mark <= stop_px * 1.0005:
                state_text = "EXITING"
                verdict = "negative" if net < 0 else "neutral"
            else:
                state_text = "HOLDING"
                verdict = "positive" if net > 0 else "negative" if net < 0 else "neutral"
            sub_text = f"{pos['pair']} {net*100:+.2f}% — entry {pos['entry']:.4g}"
        elif rendered:
            leader_sym, leader_score = max(rendered, key=lambda r: abs(r[1]))
            short = leader_sym.split("/")[0]
            state_text, verdict = _label_for_score(leader_score)
            # Add inertia annotation to the sub-text only — never let it
            # override the label/needle agreement.
            inert = inertia_view.get(leader_sym) or {}
            inertia_tag = ""
            if int(inert.get("pressure", 0)) > 0:
                inertia_tag = " (sustained buying)"
            elif int(inert.get("pressure", 0)) < 0:
                inertia_tag = " (sustained selling)"
            if state_text == "BUYING SOON":
                sub_text = f"{short} score {leader_score:+.0f}{inertia_tag}."
            elif state_text == "ALMOST":
                sub_text = f"{short} score {leader_score:+.0f}{inertia_tag}."
            elif state_text == "STAY OUT":
                sub_text = f"{short} score {leader_score:+.0f}{inertia_tag}."
            else:
                sub_text = f"Leader {short} score {leader_score:+.0f}{inertia_tag}."

        self.gauge_state.set_text(state_text)
        self.gauge_state.set_color(_verdict_color(verdict))
        self.gauge_substate.set_text(sub_text)
        self._set_panel_title(self.title_gauge, self.dot_gauge,
                              "Buy/Sell readiness — needle per coin", verdict)

    # ── [3] Hawkes predicted vs actual (per symbol) ─────────
    def _setup_hawkes(self):
        ax = self.ax_hawkes
        self._style_axes(ax)
        ax.set_xlabel("minutes ago", fontsize=8, color=COLOR_SOFT)
        ax.set_ylabel("% over next 60s", fontsize=8, color=COLOR_SOFT)
        ax.axhline(0, color="#444", linewidth=0.6, alpha=0.7)
        # Pool of (actual, predicted) line pairs per symbol slot
        self._hawk_actual_pool: list[Any] = []
        self._hawk_predicted_pool: list[Any] = []
        for i in range(8):
            la, = ax.plot([], [], linewidth=1.4, color=EXTRA_PALETTE[i % len(EXTRA_PALETTE)])
            la.set_visible(False)
            self._hawk_actual_pool.append(la)
            lp, = ax.plot([], [], linewidth=1.0, linestyle=":",
                          color=EXTRA_PALETTE[i % len(EXTRA_PALETTE)])
            lp.set_visible(False)
            self._hawk_predicted_pool.append(lp)
        self.hawk_legend = None
        self.title_hawkes = ax.set_title("", color=COLOR_TEXT, fontsize=11, loc="left")
        self.dot_hawkes = ax.text(0.005, 1.025, "●", transform=ax.transAxes,
                                  fontsize=14, color=COLOR_NEU, va="bottom", ha="left")
        self.caption_hawkes = ax.text(
            0.005, -0.22, "", transform=ax.transAxes, fontsize=8.5, color=COLOR_SOFT,
            va="top", ha="left",
        )

    def draw_hawkes(self, snap):
        ax = self.ax_hawkes
        view = snap["market"]
        n = view.ts
        multi_prices = view.multi_prices or {}
        multi_preds = view.multi_hawkes_predictions or {}
        learning = view.hawkes_learning or {}

        # Reset pools
        for ln in self._hawk_actual_pool:
            ln.set_visible(False)
            ln.set_data([], [])
        for ln in self._hawk_predicted_pool:
            ln.set_visible(False)
            ln.set_data([], [])

        # Build a stable display order: forced first, then watch/extra.
        ordered = [s for s in FORCE_SYMBOLS if s in multi_prices or s in multi_preds]
        for s in multi_preds.keys():
            if s not in ordered:
                ordered.append(s)
        ordered = ordered[:len(self._hawk_actual_pool)]

        all_xs: list[float] = []
        all_ys: list[float] = []
        legend_handles = []
        legend_labels = []
        per_symbol_hits: dict[str, tuple[int, int]] = {}  # symbol -> (agree, total)

        for i, sym in enumerate(ordered):
            color = _symbol_color(sym, i)

            # Realised 60s-forward return at each historic timestamp
            prices = list(multi_prices.get(sym) or ())
            actual_xs: list[float] = []
            actual_ys: list[float] = []
            if len(prices) >= 4:
                arr = np.asarray(prices, dtype=float)
                ts = arr[:, 0]
                px = arr[:, 1]
                for t0, p0 in prices:
                    if p0 <= 0:
                        continue
                    target = t0 + 60.0
                    j = int(np.searchsorted(ts, target))
                    if j >= len(prices):
                        break
                    p1 = float(px[j])
                    if p1 <= 0:
                        continue
                    actual_xs.append((t0 - n) / 60.0)
                    actual_ys.append((p1 / p0 - 1.0) * 100.0)

            preds = list(multi_preds.get(sym) or ())
            pred_xs = [(t - n) / 60.0 for t, _, _ in preds]
            pred_ys = [pct * 100.0 for _, pct, _ in preds]

            if not actual_xs and not pred_xs:
                continue

            la = self._hawk_actual_pool[i]
            la.set_data(actual_xs, actual_ys)
            la.set_color(color)
            la.set_visible(bool(actual_xs))

            lp = self._hawk_predicted_pool[i]
            lp.set_data(pred_xs, pred_ys)
            lp.set_color(color)
            lp.set_visible(bool(pred_xs))

            all_xs.extend(actual_xs + pred_xs)
            all_ys.extend(actual_ys + pred_ys)

            # Per-symbol hits come from the learning state (the same counts
            # used by the title aggregate, so the two can never disagree).
            lp_state = learning.get(sym) or {}
            agree = int(lp_state.get("hit_count") or 0)
            total = int(lp_state.get("n_with_move") or 0)
            per_symbol_hits[sym] = (agree, total)

            short = sym.split("/")[0]
            status = lp_state.get("status", "cold")
            n_matured = int(lp_state.get("n_matured") or 0)
            badge = _LEARNING_BADGES.get(status, status)
            if total >= 5:
                label = f"{short}  {badge}  {agree}/{total}"
            elif n_matured > 0:
                label = f"{short}  {badge}  {n_matured} samples"
            else:
                label = f"{short}  {badge}"
            legend_handles.append(la if la.get_visible() else lp)
            legend_labels.append(label)

        # Legend
        if self.hawk_legend is not None:
            try:
                self.hawk_legend.remove()
            except Exception:
                pass
            self.hawk_legend = None
        if legend_handles:
            self.hawk_legend = ax.legend(
                legend_handles, legend_labels, loc="upper left",
                fontsize=8, framealpha=0.45, labelcolor=COLOR_TEXT,
                ncol=min(len(legend_handles), 4),
            )

        # Limits
        if all_xs:
            ax.set_xlim(min(all_xs), max(max(all_xs), min(all_xs) + 0.5))
        else:
            ax.set_xlim(-5, 0)
        ax.set_ylim(*_pad_limits(all_ys + [0.0], frac=0.15, fallback=(-0.5, 0.5)))

        # Panel verdict: aggregate across symbols (weighted by sample count).
        verdict = "neutral"
        total_agree, total_n = _hawkes_hit_aggregate(per_symbol_hits)
        title_tail = ""
        if total_n >= 5:
            frac = total_agree / total_n
            if frac >= 0.6:
                verdict = "positive"
                title_tail = f"called direction {total_agree}/{total_n} right"
            elif frac <= 0.4:
                verdict = "negative"
                title_tail = f"only {total_agree}/{total_n} right"
            else:
                verdict = "neutral"
                title_tail = f"mixed {total_agree}/{total_n}"
        else:
            title_tail = "collecting outcomes"

        self._set_panel_title(self.title_hawkes, self.dot_hawkes,
                              f"Hawkes predicted vs actual — {title_tail}", verdict)
        self.caption_hawkes.set_text(
            "Solid = realised next-60s return, dotted = Hawkes prediction. "
            "Per-symbol badge: cold → learning → calibrated (≥60% directional hit rate)."
        )

    # ── [4] strategies ──────────────────────────────────────
    def _setup_strats(self):
        ax = self.ax_strats
        ax.axis("off")
        self.title_strats = ax.set_title("", color=COLOR_TEXT, fontsize=11, loc="left")
        self.dot_strats = ax.text(0.005, 1.025, "●", transform=ax.transAxes,
                                  fontsize=14, color=COLOR_NEU, va="bottom", ha="left")
        # Up to 4 rows reserved. Each row: name, equity, %, sparkline.
        self._strat_rows: list[dict[str, Any]] = []
        row_height = 0.22
        top = 0.92
        for i in range(4):
            y = top - i * row_height
            name_t = ax.text(0.01, y, "", fontsize=10, color=COLOR_TEXT,
                             transform=ax.transAxes, va="top", ha="left", weight="bold")
            eq_t = ax.text(0.32, y, "", fontsize=10, color=COLOR_TEXT,
                           transform=ax.transAxes, va="top", ha="left")
            pct_t = ax.text(0.56, y, "", fontsize=10, color=COLOR_TEXT,
                            transform=ax.transAxes, va="top", ha="left", weight="bold")
            # Sparkline inset
            spark_ax = ax.inset_axes([0.76, y - 0.16, 0.22, 0.14], transform=ax.transAxes)
            spark_ax.axis("off")
            spark_line, = spark_ax.plot([], [], color=COLOR_POS, linewidth=1.3)
            self._strat_rows.append({
                "name": name_t,
                "equity": eq_t,
                "pct": pct_t,
                "spark_ax": spark_ax,
                "spark_line": spark_line,
            })
        self.caption_strats = ax.text(
            0.01, -0.04, "", fontsize=8.5, color=COLOR_SOFT,
            transform=ax.transAxes, va="top", ha="left",
        )

    def draw_strats(self, snap):
        strategies = list(snap.get("strategies") or ())
        cap_history = list(snap["market"].capital_history)
        # Render configured strategies first, placeholders below.
        verdict = "neutral"
        primary_pct = 0.0
        for i, row in enumerate(self._strat_rows):
            if i < len(strategies):
                s = strategies[i]
                row["name"].set_text(s.get("name", "—"))
                row["name"].set_color(COLOR_TEXT)
                eq = float(s.get("equity") or 0.0)
                row["equity"].set_text(f"${eq:,.2f}")
                row["equity"].set_color(COLOR_TEXT)
                pct = float(s.get("net_return_pct") or 0.0)
                row["pct"].set_text(f"{pct:+.2f}%")
                col = COLOR_POS if pct > 0.1 else COLOR_NEG if pct < -0.1 else COLOR_NEU
                row["pct"].set_color(col)
                # Sparkline — for the active strategy use the capital history
                # we already have; placeholder strategies stay empty.
                if s.get("active") and cap_history:
                    n = snap["market"].ts
                    xs = [(t - n) / 60.0 for t, _ in cap_history]
                    ys = [c for _, c in cap_history]
                    row["spark_line"].set_data(xs, ys)
                    row["spark_line"].set_color(col)
                    row["spark_ax"].set_xlim(min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
                    row["spark_ax"].set_ylim(*_pad_limits(ys, frac=0.1))
                else:
                    row["spark_line"].set_data([], [])
                if i == 0:
                    primary_pct = pct
                    if pct > 0.1:
                        verdict = "positive"
                    elif pct < -0.1:
                        verdict = "negative"
                    else:
                        verdict = "neutral"
            else:
                row["name"].set_text("— reserved for additional strategy —")
                row["name"].set_color(COLOR_DIM)
                row["equity"].set_text("")
                row["pct"].set_text("")
                row["spark_line"].set_data([], [])

        title_text = "Strategies"
        if strategies:
            title_text = f"Strategies — primary {primary_pct:+.2f}%"
        self._set_panel_title(self.title_strats, self.dot_strats, title_text, verdict)
        self.caption_strats.set_text(
            "One row per strategy. Placeholder rows are reserved for future "
            "side-by-side comparisons."
        )

    # ── [5] status prose ────────────────────────────────────
    def _setup_text_panel(self, ax, suffix, title, mono=False, fontsize=9):
        ax.axis("off")
        title_artist = ax.set_title("", color=COLOR_TEXT, fontsize=11, loc="left")
        dot_artist = ax.text(0.005, 1.025, "●", transform=ax.transAxes,
                             fontsize=14, color=COLOR_NEU, va="bottom", ha="left")
        family = "monospace" if mono else "DejaVu Sans"
        text_artist = ax.text(0.01, 0.92, "", transform=ax.transAxes,
                              fontsize=fontsize, va="top", ha="left",
                              color=COLOR_TEXT, family=family, clip_on=True)
        setattr(self, f"title{suffix}", title_artist)
        setattr(self, f"dot{suffix}", dot_artist)
        setattr(self, f"text{suffix}", text_artist)
        # Stash a base title for panels that need to update verdict only
        setattr(self, f"baseTitle{suffix}", title)

    def draw_status(self, snap):
        cfg = self.cfg
        view = snap["market"]
        micro = view.micro or {}
        pos = snap.get("position")
        lines: list[str] = []

        # State line 1
        if pos:
            mark = view.last_price or pos["entry_signal"]
            net = mark / max(pos["entry"], 1e-9) - 1.0
            r = cfg.risk(pos["regime"])
            age = int(view.ts - pos["entry_time"])
            lines.append(
                f"Holding {pos['pair']} — currently {net*100:+.2f}%, held {age}s of max {int(r['max_hold'])}s."
            )
            stop_px = pos.get("stop_price")
            if stop_px:
                side = "profit lock" if stop_px > pos["entry"] else "loss guard"
                lines.append(f"Safety net: {side} at {stop_px:.6g}.")
            lines.append(f"Why we entered: {_plain_regime(pos['regime'])} — {pos.get('note','')[:80]}")
            verdict = "positive" if net > 0 else "negative" if net < 0 else "neutral"
        else:
            lines.append("Hydra is watching, not holding any coin right now.")
            # Top blockers from candidate gating
            blockers = []
            buy_not = float(micro.get("buy_not") or 0.0)
            burst = float(micro.get("burst_ratio") or 0.0)
            move_pct = float(micro.get("move_pct") or 0.0) * 100.0
            hawkes_exc = float(micro.get("hawkes_excitation") or 0.0)
            spread = float(micro.get("spread_bps") or 0.0)
            max_spread = float(micro.get("max_spread") or 80.0)
            if buy_not < cfg.micro_min_buy_notional:
                blockers.append("not enough buying pressure")
            if burst < cfg.micro_burst_multiple:
                blockers.append("activity is normal, not a burst")
            if move_pct < cfg.micro_min_move_pct * 100.0:
                blockers.append("price hasn't moved enough")
            if cfg.hawkes_enabled and hawkes_exc < cfg.hawkes_min_excitation:
                blockers.append("crowd-wave signal is weak")
            if spread > max_spread:
                blockers.append("spread is too expensive")
            if blockers:
                lines.append("Missing for a trade: " + "; ".join(blockers[:3]) + ".")
            else:
                lines.append("Setup is close — checklist mostly green.")
            verdict = "neutral"

        # BTC backdrop — uses inertia so we don't flicker between supportive
        # and weak each time the 5m return wiggles around the threshold.
        backdrop = int(snap["market"].btc_backdrop_state)
        if snap.get("btc_flush"):
            lines.append("Market backdrop: BTC is dumping — danger mode, very picky.")
            verdict = "negative"
        elif backdrop > 0:
            lines.append("Market backdrop: BTC supportive — direction has held.")
        elif backdrop < 0:
            lines.append("Market backdrop: BTC weak — bearish bias is sticking.")
        else:
            lines.append("Market backdrop: BTC indecisive — no sustained direction.")

        # Sustained per-symbol pressure (top one).
        inertia = snap["market"].inertia_state or {}
        sustained = []
        for sym, st in inertia.items():
            p = int(st.get("pressure", 0))
            if p != 0:
                tag = "buying" if p > 0 else "selling"
                sustained.append(f"{sym.split('/')[0]} {tag}")
        if sustained:
            lines.append("Sustained pressure: " + ", ".join(sustained[:4]) + ".")

        # Exposure
        heat = 0.0
        if pos and snap.get("equity"):
            heat = pos["qty"] * (view.last_price or pos["entry_signal"]) / snap["equity"]
        lines.append(f"Wallet at work: {heat:.1%} of equity, limit {cfg.max_portfolio_heat:.0%}.")

        # Watchlist
        if snap.get("watch"):
            ws_bits = []
            for s, w in list(snap["watch"])[:4]:
                ws_bits.append(f"{s.split('/')[0]} +{w['spike_pct']:.1f}%")
            lines.append("Recently spiked: " + ", ".join(ws_bits) + ".")

        # Tuning
        opt = snap.get("optuna") or {}
        if opt.get("running"):
            done = opt.get("completed", 0)
            total = opt.get("total", 0)
            lines.append(f"Auto-tuning Optuna: {done}/{total or '?'} trials running.")
        else:
            auto = snap.get("auto") or {}
            if auto:
                lines.append(
                    f"Tuning watch: drawdown {auto.get('dd_pct',0):.2f}%, "
                    f"idle {auto.get('idle_min',0):.1f}m, missed setups {auto.get('missed',0)}."
                )

        self.text_status.set_text("\n".join(lines))
        self._set_panel_title(self.title_status, self.dot_status,
                              "What Hydra is doing", verdict)

    # ── [6] recent trades ───────────────────────────────────
    def draw_trades(self, snap):
        trades = list(snap.get("trades") or ())
        if not trades:
            self.text_trades.set_text("No closed trades yet. Hydra is waiting for a clean setup.")
            self._set_panel_title(self.title_trades, self.dot_trades,
                                  "Recent trades", "neutral")
            return
        recent = trades[-5:][::-1]
        lines: list[str] = []
        wins = 0
        for t in recent:
            etime = str(t.get("exit_time", ""))[11:19]
            sym = str(t.get("pair", "?")).split("/")[0]
            ret = float(t.get("return_pct") or 0.0)
            hold_min = float(t.get("hold_sec") or 0.0) / 60.0
            regime_ph = _plain_regime(t.get("regime", ""))
            reason_ph = _plain_reason(t.get("reason", ""))
            mark = "✓" if ret > 0 else "✗"
            if ret > 0:
                wins += 1
            verb = "made" if ret > 0 else "lost"
            lines.append(
                f"{mark} {etime}  {sym}  — {regime_ph}, {reason_ph}; "
                f"{verb} {abs(ret):.2f}% in {hold_min:.1f}m"
            )
        self.text_trades.set_text("\n".join(lines))
        n = len(recent)
        if wins > n / 2:
            verdict = "positive"
        elif wins < n / 2:
            verdict = "negative"
        else:
            verdict = "neutral"
        self._set_panel_title(self.title_trades, self.dot_trades,
                              f"Recent trades — {wins}/{n} wins", verdict)

    # ── main loop ───────────────────────────────────────────
    def update(self, _frame):
        opt = self.optuna.snapshot() if self.optuna is not None else {}
        snap = self.engine.dashboard_snapshot(
            focus=self.locked_focus, mode="LIVE",
            optuna=opt, params_source=self.params_source,
        )
        self.focus = snap["focus"]
        try:
            self.draw_chart(snap)
            self.draw_gauge(snap)
            self.draw_hawkes(snap)
            self.draw_strats(snap)
            self.draw_status(snap)
            self.draw_trades(snap)
        except Exception as e:
            print(f"  [DASH] draw error: {e}")
            import traceback
            traceback.print_exc()
        return []

    def show(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=200, cache_frame_data=False)
        plt.show()
