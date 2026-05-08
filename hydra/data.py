from __future__ import annotations

import glob
import gzip
import json
import os
import time
import zlib
from pathlib import Path
from typing import Any, Iterable

import requests

from .constants import BOOK_N, BTC_SYMBOL, FALLBACK_SYMBOLS, FORCE_SYMBOLS, REST, TOP_N
from .utils import sf


def get_universe(http: requests.Session | None = None) -> tuple[list[str], list[str], list[str]]:
    http = http or requests.Session()
    print("Fetching universe from Kraken REST API...")
    for attempt in range(3):
        try:
            p = http.get(f"{REST}/AssetPairs", timeout=20)
            p.raise_for_status()
            pairs = p.json()["result"]
            t = http.get(f"{REST}/Ticker", timeout=20)
            t.raise_for_status()
            ticker = t.json()["result"]
            break
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(2)
    else:
        print("Using fallback universe.")
        fallback = list(dict.fromkeys(FALLBACK_SYMBOLS + FORCE_SYMBOLS))
        return fallback, list(dict.fromkeys(FORCE_SYMBOLS)), list(dict.fromkeys(FORCE_SYMBOLS))

    usd = {k: v for k, v in pairs.items() if v.get("quote") in ("ZUSD", "USD") and v.get("status") == "online"}
    ranked = []
    for k, info in usd.items():
        wsname = info.get("wsname")
        if not wsname or k not in ticker:
            continue
        last = sf(ticker[k]["c"][0])
        v24 = sf(ticker[k]["v"][1])
        if last > 0 and v24 > 0:
            ranked.append((wsname, last * v24))
    ranked.sort(key=lambda x: -x[1])
    chosen = [w for w, _ in ranked[:TOP_N]]
    for s in FORCE_SYMBOLS:
        if s not in chosen:
            chosen.append(s)
    if BTC_SYMBOL not in chosen:
        chosen.append(BTC_SYMBOL)
    chosen = list(dict.fromkeys(chosen))
    book_syms = list(dict.fromkeys([w for w, _ in ranked[:BOOK_N]] + FORCE_SYMBOLS))
    book_syms = [s for s in book_syms if s in chosen]
    trade_syms = list(book_syms)
    return chosen, book_syms, trade_syms


def looks_like_gzip(path: str | os.PathLike) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return str(path).endswith(".gz")


def gzip_is_complete(path: str | os.PathLike) -> bool:
    if not looks_like_gzip(path):
        return True
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (EOFError, zlib.error, OSError):
        return False


def open_recorder(path: str | os.PathLike):
    path = str(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path) and looks_like_gzip(path) and not gzip_is_complete(path):
        rotated = f"{path}.broken-{int(time.time())}"
        os.replace(path, rotated)
        print(f"[RECORDER] rotated broken gzip to {rotated}")
    if path.endswith(".gz"):
        return gzip.open(path, "at", encoding="utf-8")
    return open(path, "a", encoding="utf-8")


def open_replay(path: str | os.PathLike, allow_partial: bool = True):
    path = str(path)
    if looks_like_gzip(path):
        # gzip.open raises EOFError on close/read if truncated. replay_messages
        # catches it and returns the readable prefix.
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


_warned_partial: set[str] = set()


def replay_messages(path: str | os.PathLike, allow_partial: bool = True) -> Iterable[tuple[float, dict[str, Any] | None]]:
    path = str(path)
    try:
        with open_replay(path, allow_partial=allow_partial) as f:
            while True:
                try:
                    line = f.readline()
                except (EOFError, zlib.error) as e:
                    if allow_partial:
                        if path not in _warned_partial:
                            print(f"[REPLAY] warning: using readable prefix of truncated gzip {path}: {e}")
                            _warned_partial.add(path)
                        break
                    raise
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield float(rec.get("ts", 0.0)), rec.get("msg")
    except (EOFError, zlib.error) as e:
        if allow_partial:
            if path not in _warned_partial:
                print(f"[REPLAY] warning: using readable prefix of truncated gzip {path}: {e}")
                _warned_partial.add(path)
            return
        raise


def count_replay_messages(path: str | os.PathLike, max_records: int | None = None) -> tuple[int, float | None, float | None]:
    n = 0
    first = last = None
    for t, _ in replay_messages(path, allow_partial=True):
        if first is None:
            first = t
        last = t
        n += 1
        if max_records is not None and n >= max_records:
            break
    return n, first, last


def recording_search_roots() -> list[str]:
    roots = ["runs", "."]
    home_runs = Path.home() / "runs"
    if home_runs.exists():
        roots.append(str(home_runs))
    return roots


def list_recordings(limit: int = 8, roots: list[str] | None = None) -> list[str]:
    roots = roots or recording_search_roots()
    paths: list[str] = []
    for root in roots:
        for pattern in ("*.jsonl", "*.jsonl.gz", "*.gz"):
            paths.extend(glob.glob(os.path.join(root, pattern)))
    paths = list(dict.fromkeys(paths))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[:limit]


def filter_paths_for_shared_read(paths: list[str], active_record_path: str | None = None, min_records: int = 10) -> list[str]:
    out: list[str] = []
    active = os.path.normpath(os.path.abspath(active_record_path)) if active_record_path else None
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        ap = os.path.normpath(os.path.abspath(p))
        if active and ap == active:
            continue
        try:
            if os.path.getsize(p) <= 0:
                continue
            n, first, last = count_replay_messages(p, max_records=min_records)
            if n <= 0:
                continue
            out.append(p)
        except Exception as e:
            print(f"[REPLAY] skip unreadable recording {p}: {e}")
    return out


def infer_recording_symbols(paths: list[str], max_lines_per_file: int = 200_000) -> tuple[list[str], list[str], list[str]]:
    ohlc: list[str] = []
    book: list[str] = []
    trade: list[str] = []

    def add(target: list[str], s: str | None) -> None:
        if s and s not in target:
            target.append(s)

    for path in paths:
        seen = 0
        for _, msg in replay_messages(path, allow_partial=True):
            seen += 1
            if seen > max_lines_per_file:
                break
            if not isinstance(msg, dict):
                continue
            ch = msg.get("channel")
            data = msg.get("data") or []
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                s = row.get("symbol")
                if ch == "ohlc":
                    add(ohlc, s)
                elif ch == "book":
                    add(book, s)
                elif ch == "trade":
                    add(trade, s)
    for s in FORCE_SYMBOLS:
        add(ohlc, s)
        add(book, s)
        add(trade, s)
    if BTC_SYMBOL not in ohlc:
        ohlc.append(BTC_SYMBOL)
    return ohlc, book, trade



