from __future__ import annotations

import json
import time
from typing import Any

import requests

from .clock import Clock, VirtualClock
from .config import Config
from .constants import FORCE_SYMBOLS, REST
from .data import infer_recording_symbols, replay_messages
from .engine import HydraEngine


_wsname_to_pair_cache: dict[str, str] = {}


def run_replay_backtest(replay_path: str, cfg: Config | None = None, verbose: bool = False) -> dict[str, Any]:
    cfg = cfg or Config()
    paths = [replay_path]
    symbols, book_syms, trade_syms = infer_recording_symbols(paths)
    clock = Clock()
    vclock = VirtualClock()
    clock.set(vclock)
    engine = HydraEngine(
        cfg,
        symbols=symbols,
        book_symbols=book_syms,
        trade_symbols=trade_syms,
        clock=clock,
        name="HydraBT",
        csv_enabled=False,
        verbose=verbose,
    )

    n = 0
    last_capital_sample = 0.0
    for t, m in replay_messages(replay_path, allow_partial=True):
        vclock.advance_to(t)
        if m is None:
            continue
        engine._process_message_unguarded(m)
        n += 1
        if t - last_capital_sample >= 1.0:
            engine.sample_capital()
            last_capital_sample = t
    engine.force_exit_eob()
    metrics = engine.metrics()
    metrics["messages"] = n
    if verbose:
        print(json.dumps(metrics, indent=2, default=str))
    return metrics


def _wsname_to_pair(wsname: str, http: requests.Session | None = None) -> str:
    if wsname in _wsname_to_pair_cache:
        return _wsname_to_pair_cache[wsname]
    http = http or requests.Session()
    r = http.get(f"{REST}/AssetPairs", timeout=20)
    r.raise_for_status()
    pairs = r.json()["result"]
    candidates = [wsname]
    if "BTC" in wsname:
        candidates.append(wsname.replace("BTC", "XBT"))
    elif "XBT" in wsname:
        candidates.append(wsname.replace("XBT", "BTC"))
    altname = wsname.replace("/", "")
    for k, info in pairs.items():
        if info.get("wsname") in candidates or info.get("altname") == altname:
            _wsname_to_pair_cache[wsname] = k
            return k
    raise RuntimeError(f"unknown pair: {wsname}")


def fetch_kraken_ohlc(wsname: str, interval: int = 1, since: int | None = None, http: requests.Session | None = None) -> list[list[Any]]:
    http = http or requests.Session()
    pair = _wsname_to_pair(wsname, http=http)
    params: dict[str, Any] = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since
    r = http.get(f"{REST}/OHLC", params=params, timeout=20)
    r.raise_for_status()
    res = r.json()
    if res.get("error"):
        raise RuntimeError(f"Kraken error: {res['error']}")
    out = res["result"]
    pair_key = next(k for k in out if k != "last")
    return out[pair_key]


def fetch_ohlc_bars_for_symbols(symbols: list[str], interval: int = 1, hours: int = 12) -> list[tuple]:
    http = requests.Session()
    all_bars = []
    since = int(time.time() - hours * 3600 - 3600)
    for sym in symbols:
        try:
            bars = fetch_kraken_ohlc(sym, interval=interval, since=since, http=http)
            for bar in bars:
                t = float(bar[0])
                all_bars.append((t, sym, float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), float(bar[6])))
        except Exception as e:
            print(f"[HISTORICAL] fetch {sym} failed: {e}")
    if not all_bars:
        return []
    all_bars.sort(key=lambda x: x[0])
    cutoff = all_bars[-1][0] - hours * 3600
    return [b for b in all_bars if b[0] >= cutoff]


def replay_ohlc_bars(all_bars: list[tuple], cfg: Config, name: str = "HydraOHLC", verbose: bool = False, sleep_per_bar: float = 0.0) -> HydraEngine:
    symbols = list(dict.fromkeys([b[1] for b in all_bars] + FORCE_SYMBOLS))
    clock = Clock()
    vclock = VirtualClock()
    clock.set(vclock)
    # Candle-only mode intentionally bypasses microstructure gates, but the
    # engine still disables book_ignition by never receiving trade/book events.
    cfg = dataclasses_replace_macro_for_candle(cfg)
    engine = HydraEngine(cfg, symbols=symbols, book_symbols=[], trade_symbols=[], clock=clock, name=name, csv_enabled=False, verbose=verbose, ohlc_only=True)
    seen_syms = set()
    last_cap_t = 0.0
    for t, sym, o, h, l, c, vol in all_bars:
        vclock.advance_to(t)
        msg_type = "snapshot" if sym not in seen_syms else "update"
        seen_syms.add(sym)
        engine._process_message_unguarded({
            "channel": "ohlc",
            "type": msg_type,
            "data": [{"symbol": sym, "interval_begin": str(int(t)), "open": o, "high": h, "low": l, "close": c, "volume": vol}],
        })
        if t - last_cap_t >= 60.0:
            engine.sample_capital()
            last_cap_t = t
        if sleep_per_bar > 0:
            time.sleep(sleep_per_bar)
    engine.force_exit_eob()
    if all_bars:
        engine.sample_capital()
    return engine


def dataclasses_replace_macro_for_candle(cfg: Config) -> Config:
    # Avoid importing dataclasses at module top solely for this tiny helper? Keep
    # it local to make the candle-only behavior explicit.
    import dataclasses

    out = dataclasses.replace(cfg)
    out.macro_require_microstructure = False
    return out


def run_historical_backtest(symbols: list[str] | None = None, interval: int = 1, hours: int = 12, cfg: Config | None = None, verbose: bool = True) -> dict[str, Any]:
    cfg = cfg or Config()
    symbols = symbols or list(FORCE_SYMBOLS)
    print("HISTORICAL CANDLE TEST")
    print("  source: Kraken REST OHLC candles")
    print("  book_ignition: disabled (no trade tape or L2 book in REST OHLC)")
    print("  live execution: approximate; use websocket recording replay for full strategy validation")
    print(f"  symbols: {', '.join(symbols)}  hours={hours} interval={interval}m")
    all_bars = fetch_ohlc_bars_for_symbols(symbols, interval=interval, hours=hours)
    if not all_bars:
        raise RuntimeError("no historical OHLC bars fetched")
    engine = replay_ohlc_bars(all_bars, cfg, verbose=False)
    metrics = engine.metrics()
    if verbose:
        print(json.dumps(metrics, indent=2, default=str))
    return metrics
