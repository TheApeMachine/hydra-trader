from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from typing import Any

import requests
import websockets

from .clock import Clock, VirtualClock
from .config import Config
from .constants import FORCE_SYMBOLS, STATUS_EVERY, WS
from .data import get_universe, infer_recording_symbols, open_recorder, replay_messages
from .market import MarketStore
from .trader import HydraTrader, TraderHooks
from .utils import breakeven_move_pct, sf


class RunControl:
    """Shared runtime flags for live UI/optimizer/backtest coordination."""

    def __init__(self):
        self.live_paused = threading.Event()
        self.stop_requested = threading.Event()
        self.active_record_path: str | None = None


class AutoStats:
    """Small first-class Optuna trigger state, decoupled from Trader internals."""

    def __init__(self, clock):
        self.clock = clock
        self.lock = threading.Lock()
        self.rejected = 0
        self.expired = 0
        self.peak_equity: float | None = None
        self.last_fill_ts: float | None = None
        self.session_start_ts: float | None = None

    def boot(self) -> None:
        with self.lock:
            self.rejected = 0
            self.expired = 0
            self.peak_equity = None
            self.last_fill_ts = None
            self.session_start_ts = self.clock.now()

    def candidate_expired(self, n: int) -> None:
        if n <= 0:
            return
        with self.lock:
            self.expired += n

    def candidate_rejected(self, n: int) -> None:
        if n <= 0:
            return
        with self.lock:
            self.rejected += n

    def fill(self) -> None:
        with self.lock:
            self.last_fill_ts = self.clock.now()

    def snapshot(self, equity: float) -> dict[str, float | int | None]:
        with self.lock:
            if self.peak_equity is None or equity > self.peak_equity:
                self.peak_equity = equity
            peak = self.peak_equity
            last_fill = self.last_fill_ts
            sess0 = self.session_start_ts
            idle_min = 0.0
            if last_fill is not None:
                idle_min = max(0.0, (self.clock.now() - last_fill) / 60.0)
            elif sess0 is not None:
                idle_min = max(0.0, (self.clock.now() - sess0) / 60.0)
            dd_pct = ((peak - equity) / peak * 100.0) if peak and peak > 0 else 0.0
            return {
                "rejected": self.rejected,
                "expired": self.expired,
                "missed": self.rejected + self.expired,
                "peak_equity": peak,
                "idle_min": idle_min,
                "dd_pct": dd_pct,
            }


class HydraEngine:
    """Owns a complete independent strategy runtime.

    Every live session, replay backtest, historical candle test, and Optuna trial
    creates its own HydraEngine. No strategy state is global anymore.
    """

    def __init__(
        self,
        cfg: Config,
        symbols: list[str] | None = None,
        book_symbols: list[str] | None = None,
        trade_symbols: list[str] | None = None,
        clock: Clock | None = None,
        name: str = "Hydra",
        csv_enabled: bool = True,
        verbose: bool = True,
        ohlc_only: bool = False,
        run_control: RunControl | None = None,
        auto_stats: AutoStats | None = None,
    ):
        self.cfg = cfg
        self.clock = clock or Clock()
        self.symbols = list(symbols or FORCE_SYMBOLS)
        self.book_symbols = list(book_symbols or FORCE_SYMBOLS)
        self.trade_symbols = list(trade_symbols or FORCE_SYMBOLS)
        self.market = MarketStore(cfg, self.clock, self.book_symbols, self.trade_symbols, ohlc_only=ohlc_only)
        self.run_control = run_control or RunControl()
        self.auto_stats = auto_stats or AutoStats(self.clock)
        hooks = TraderHooks(
            candidate_expired=self.auto_stats.candidate_expired,
            candidate_rejected=self.auto_stats.candidate_rejected,
            fill=self.auto_stats.fill,
        )
        self.trader = HydraTrader(cfg, self.market, self.clock, name=name, csv_enabled=csv_enabled, verbose=verbose, hooks=hooks)
        self.verbose = verbose
        self.n_messages = 0

    def set_symbols(self, symbols: list[str], book_symbols: list[str], trade_symbols: list[str]) -> None:
        self.symbols = list(symbols)
        self.book_symbols = list(book_symbols)
        self.trade_symbols = list(trade_symbols)
        self.market.set_symbols(book_symbols=self.book_symbols, trade_symbols=self.trade_symbols)

    def capital(self) -> float:
        return self.trader.capital_now() if self.trader.pos else self.trader.cash

    def process_message(self, m: dict[str, Any], live_gate: bool = True) -> None:
        if live_gate and self.run_control.live_paused.is_set():
            return
        self._process_message_unguarded(m)

    def _process_message_unguarded(self, m: dict[str, Any]) -> None:
        if not isinstance(m, dict):
            return
        channel = m.get("channel")
        data = m.get("data", [])
        if not isinstance(data, list):
            return
        self.n_messages += 1

        if channel == "ohlc":
            msg_type = m.get("type", "")
            if msg_type == "snapshot":
                for c in data:
                    event = self.market.update_candle(c)
                    s = c.get("symbol")
                    if s:
                        self.market.symbol_live[s] = True
                return
            if msg_type == "update":
                for c in data:
                    s = c.get("symbol")
                    if s and not self.market.symbol_live[s]:
                        self.market.symbol_live[s] = True
                    event = self.market.update_candle(c)
                    if event and self.market.symbol_live[event.symbol]:
                        self.trader.on_closed_candle(event.symbol, event.candle, event.minute_key)
                    if s:
                        self.trader.on_tick(s, sf(c.get("close")))
                self.trader.execute_due()

        elif channel == "trade" and m.get("type") == "update":
            for tr in data:
                sym = tr.get("symbol")
                if not sym or not self.market.symbol_live.get(sym):
                    continue
                price = sf(tr.get("price"))
                qty = sf(tr.get("qty"))
                side = tr.get("side")
                if price <= 0 or qty <= 0 or side not in ("buy", "sell"):
                    continue
                self.market.add_trade_to_tape(sym, side, price, qty)
                self.trader.on_tick(sym, price)
                self.trader.maybe_micro_candidate(sym)
            self.trader.execute_due()

        elif channel == "book" and m.get("type") in ("snapshot", "update"):
            msg_type = m.get("type")
            for b in data:
                sym = b.get("symbol")
                if not sym:
                    continue
                self.market.apply_book(sym, msg_type, b.get("bids", []), b.get("asks", []))
                self.trader.on_book(sym)
            self.trader.execute_due()

    def sample_capital(self) -> None:
        self.market.sample_capital(self.clock.now(), self.capital())

    def force_exit_eob(self) -> None:
        if self.trader.pos:
            pair = self.trader.pos["pair"]
            mark = self.market.mark_for_exit(pair) or self.trader.pos["entry_signal"]
            self.trader._exit(mark, "eob")

    def metrics(self) -> dict[str, Any]:
        return self.trader.summary_metrics(self.n_messages)

    def summary_print(self) -> None:
        final = self.capital()
        wins = sum(1 for t in self.trader.trades if t["pnl_usd"] > 0)
        total = len(self.trader.trades)
        print("\n" + "=" * 64)
        print(f"\n[{self.trader.name}] final=${final:.4f} trades={total} wins={wins} losses={total - wins}")
        if self.trader.trades:
            pnls = [t["pnl_usd"] for t in self.trader.trades]
            print(f"   best=${max(pnls):+.4f}  worst=${min(pnls):+.4f}  total=${sum(pnls):+.4f}")
            by_regime: dict[str, list[dict[str, Any]]] = {}
            for t in self.trader.trades:
                by_regime.setdefault(t["regime"], []).append(t)
            for regime, trs in sorted(by_regime.items()):
                rpnl = sum(x["pnl_usd"] for x in trs)
                rw = sum(1 for x in trs if x["pnl_usd"] > 0)
                print(f"   {regime}: {len(trs)} trades, {rw} wins, pnl=${rpnl:+.4f}")
            print(f"   log: {self.trader.csv_path}")
        print("=" * 64 + "\n")

    def dashboard_snapshot(self, focus: str | None = None, mode: str = "LIVE", optuna: dict | None = None, backtest: dict | None = None, params_source: str = "defaults") -> dict[str, Any]:
        focus_sym = self.market.focus_candidate(focus, self.trader.pos["pair"] if self.trader.pos else None)
        view = self.market.dashboard_view(focus_sym)
        auto = self.auto_stats.snapshot(self.capital())
        return {
            "mode": mode,
            "params_source": params_source,
            "market": view,
            "focus": focus_sym,
            "cash": self.trader.cash,
            "equity": self.capital(),
            "position": dict(self.trader.pos) if self.trader.pos else None,
            "trades": tuple(self.trader.trades),
            "watch": tuple((s, dict(w)) for s, w in self.trader.watch.items()),
            "candidates": tuple(dict(c) for c in self.trader.candidates.values()),
            "btc_ok": self.market.btc_context_ok(False),
            "btc_strict": self.market.btc_context_ok(True),
            "btc_flush": self.market.btc_flush_active(),
            "auto": auto,
            "optuna": optuna or {},
            "backtest": backtest or {},
        }


async def subscribe_in_chunks(ws, channel: str, symbols: list[str], chunk: int = 50, extra: dict | None = None) -> None:
    extra = extra or {}
    for i in range(0, len(symbols), chunk):
        await ws.send(json.dumps({
            "method": "subscribe",
            "params": {"channel": channel, "symbol": symbols[i:i + chunk], **extra},
        }))
        await asyncio.sleep(0.4)


async def ws_session(engine: HydraEngine, recorder=None) -> None:
    last_flush_wall = 0.0
    async with websockets.connect(WS, ping_interval=20, max_size=2 ** 23) as ws:
        for s in engine.symbols:
            engine.market.symbol_live[s] = False
        await subscribe_in_chunks(ws, "ohlc", engine.symbols, chunk=50, extra={"interval": 1, "snapshot": True})
        await subscribe_in_chunks(ws, "trade", engine.trade_symbols, chunk=50, extra={"snapshot": False})
        await subscribe_in_chunks(ws, "book", engine.book_symbols, chunk=50, extra={"depth": 10, "snapshot": True})
        print("Connected. Ingesting snapshots...\n")
        async for msg in ws:
            try:
                m = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if recorder is not None:
                wall = time.time()
                recorder.write(json.dumps({"ts": wall, "msg": m}) + "\n")
                if wall - last_flush_wall >= 1.0:
                    recorder.flush()
                    last_flush_wall = wall
            engine.process_message(m)


async def status_loop(engine: HydraEngine) -> None:
    while not engine.run_control.stop_requested.is_set():
        await asyncio.sleep(STATUS_EVERY)
        if engine.run_control.live_paused.is_set():
            continue
        engine.trader.cleanup()
        engine.trader.execute_due()
        t = time.strftime("%H:%M:%S")
        if engine.trader.pos:
            p = engine.trader.pos
            cur = engine.market.mark_for_exit(p["pair"]) or p["entry_signal"]
            cap = engine.trader.capital_now(cur)
            net = cap / max(p["entry_cost"], 1e-9) - 1.0
            age = int(engine.clock.now() - p["entry_time"])
            sp = engine.market.spread_bps(p["pair"])
            imb = engine.market.book_imbalance(p["pair"], 5)
            book_txt = f" | sp {sp:.1f}bps book {imb:.2f}" if sp is not None and imb is not None else ""
            print(f"  [{t}] ${cap:.4f} | {p['pair']} [{p['regime']}] | net {net*100:+.2f}% | Trail {'ON' if p['trail_active'] else 'OFF'} | Hold {age}s | Watch {len(engine.trader.watch)} | Cand {len(engine.trader.candidates)}{book_txt}")
        else:
            print(f"  [{t}] ${engine.trader.cash:.4f} | FLAT | Trades {len(engine.trader.trades)} | Watch {len(engine.trader.watch)} | Cand {len(engine.trader.candidates)}")


async def capital_tracker(engine: HydraEngine) -> None:
    while not engine.run_control.stop_requested.is_set():
        await asyncio.sleep(1.0)
        if engine.run_control.live_paused.is_set():
            continue
        engine.sample_capital()


async def run_live(engine: HydraEngine, record_path: str | None = None) -> None:
    recorder = open_recorder(record_path) if record_path else None
    engine.run_control.active_record_path = os.path.abspath(record_path) if record_path else None
    if recorder:
        print(f"Recording WS messages to {record_path}\n")
    asyncio.create_task(status_loop(engine))
    asyncio.create_task(capital_tracker(engine))
    try:
        while not engine.run_control.stop_requested.is_set():
            try:
                await ws_session(engine, recorder=recorder)
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                print(f"  [WS] Lost: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"  [WS] Error: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
    finally:
        if recorder:
            recorder.close()
        engine.run_control.active_record_path = None


def start_ws_thread(engine: HydraEngine, record_path: str | None = None) -> threading.Thread:
    def runner():
        try:
            asyncio.run(run_live(engine, record_path=record_path))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"  [WS THREAD] fatal: {e}")
    th = threading.Thread(target=runner, name="hydra-ws", daemon=True)
    th.start()
    return th


def start_replay_thread(engine: HydraEngine, replay_path: str, speed: float) -> threading.Thread:
    vclock = VirtualClock()
    engine.clock.set(vclock)

    def runner():
        try:
            wallclock_start = time.time()
            virtual_start = None
            for t, m in replay_messages(replay_path):
                while engine.run_control.live_paused.is_set():
                    time.sleep(0.05)
                if virtual_start is None:
                    virtual_start = t
                vclock.advance_to(t)
                if m is None:
                    continue
                engine.process_message(m)
                if speed != float("inf"):
                    target_wall = wallclock_start + (t - virtual_start) / speed
                    delay = target_wall - time.time()
                    if delay > 0:
                        time.sleep(delay)
            print("\n[REPLAY] end of file. Strategy state preserved; close window for summary.")
            engine.force_exit_eob()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"  [REPLAY THREAD] fatal: {e}")
    th = threading.Thread(target=runner, name="hydra-replay", daemon=True)
    th.start()
    return th


def start_capital_tracker_thread(engine: HydraEngine) -> threading.Thread:
    def runner():
        while not engine.run_control.stop_requested.is_set():
            time.sleep(1.0)
            if engine.run_control.live_paused.is_set():
                continue
            engine.sample_capital()
    th = threading.Thread(target=runner, name="hydra-cap", daemon=True)
    th.start()
    return th


def build_live_engine(cfg: Config, params_source: str, record_or_replay_paths: list[str] | None = None, ohlc_only: bool = False) -> HydraEngine:
    if record_or_replay_paths:
        symbols, book_syms, trade_syms = infer_recording_symbols(record_or_replay_paths)
    else:
        symbols, book_syms, trade_syms = get_universe(requests.Session())
    engine = HydraEngine(cfg, symbols=symbols, book_symbols=book_syms, trade_symbols=trade_syms, name="Hydra", csv_enabled=True, verbose=True, ohlc_only=ohlc_only)
    print(
        f"Watching {len(symbols)} OHLC pairs + {len(engine.trade_symbols)} trade pairs + {len(engine.book_symbols)} L2 book pairs. Capital: ${cfg.start_capital:.2f}"
    )
    print(f"Breakeven fallback move under fee/slippage model: {breakeven_move_pct(cfg)*100:.2f}%")
    print(f"Book/trade symbols: {', '.join(engine.trade_symbols[:40])}{'...' if len(engine.trade_symbols) > 40 else ''}")
    print(f"Params: {params_source}\n")
    return engine
