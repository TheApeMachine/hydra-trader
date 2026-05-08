from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .backtest import run_historical_backtest, run_replay_backtest
from .config import maybe_load_config
from .dashboard import Dashboard
from .engine import build_live_engine, start_capital_tracker_thread, start_replay_thread, start_ws_thread
from .optimizer import AutoOptunaWatcher, OPTUNA_AVAILABLE, OptunaController


EXAMPLES = """\
examples:
  # Live trading with dashboard; auto-loads ./hydra_best.json if present:
  python master.py

  # Live + record every WS message to a plain JSONL file:
  python master.py --record runs/today.jsonl --no-params

  # Replay a closed recording through the dashboard at 10× speed:
  python master.py --replay runs/today.jsonl --speed 10 --no-params

  # Full-signal headless replay backtest from a WS recording:
  python master.py --replay runs/today.jsonl --backtest --no-params

  # Simple historical candle-only test from Kraken REST OHLC:
  python master.py --historical-backtest --hist-symbols BTC/USD,ETH/USD,SOL/USD --hist-hours 12 --no-params

  # Headless Optuna on closed recording(s); writes hydra_candidate.json and maybe hydra_best.json:
  python master.py --optuna --optuna-recordings runs/day1.jsonl,runs/day2.jsonl --optuna-trials 100 --no-params

  # Experimental auto-Optuna in simulated live mode:
  python master.py --record runs/today.jsonl --auto-optuna --no-params
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Hydra v5 modular paper trader: live dashboard, recorder, replay, historical test, Optuna.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--record", metavar="PATH", help="Live mode: append every Kraken WS message to PATH (.jsonl recommended).")
    p.add_argument("--replay", metavar="PATH", help="Replay a closed recording instead of connecting to Kraken WS.")
    p.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier. Use inf for max speed.")
    p.add_argument("--backtest", action="store_true", help="With --replay: run headless full-signal replay backtest.")
    p.add_argument("--historical-backtest", action="store_true", help="Run Kraken REST OHLC candle-only historical test.")
    p.add_argument("--hist-symbols", default="BTC/USD,ETH/USD,SOL/USD", help="Comma-separated symbols for --historical-backtest.")
    p.add_argument("--hist-hours", type=int, default=12, help="Hours for --historical-backtest (Kraken OHLC is limited to recent candles).")
    p.add_argument("--hist-interval", type=int, default=1, help="Kraken OHLC interval in minutes for --historical-backtest.")
    p.add_argument("--no-dashboard", action="store_true", help="Run live/replay without opening the dashboard window.")
    p.add_argument("--params", metavar="PATH", help="Load tuned params from JSON. Default: auto-load ./hydra_best.json if it exists.")
    p.add_argument("--no-params", action="store_true", help="Ignore saved tuned params; use Config defaults.")

    p.add_argument("--optuna", action="store_true", help="Run Optuna headlessly and exit.")
    p.add_argument("--optuna-recordings", default="", help="Comma-separated closed recording paths for Optuna.")
    p.add_argument("--optuna-trials", type=int, default=100, help="Trials for manual/headless Optuna.")
    p.add_argument("--optuna-train-frac", type=float, default=0.7, help="Train/validation split fraction for Optuna.")

    g = p.add_argument_group("experimental auto Optuna")
    g.add_argument("--auto-optuna", action="store_true", help="Start Optuna automatically on simulated degradation triggers.")
    g.add_argument("--auto-opt-dd-pct", type=float, default=5.0, help="Fire when equity drops this %% below running peak.")
    g.add_argument("--auto-opt-idle-min", type=float, default=45.0, help="Flat no-fill minutes for idle trigger.")
    g.add_argument("--auto-opt-missed", type=int, default=8, help="Flat+idle missed candidate threshold.")
    g.add_argument("--auto-opt-cooldown-min", type=float, default=60.0, help="Minutes after any auto study before firing again.")
    g.add_argument("--auto-opt-trials", type=int, default=100, help="Trials per auto-started study.")
    return p.parse_args(argv)


def _parse_recordings(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    args = parse_args(argv)
    cfg, params_source = maybe_load_config(args.params, no_params=args.no_params)

    if args.historical_backtest:
        symbols = _parse_recordings(args.hist_symbols)
        run_historical_backtest(symbols=symbols, interval=args.hist_interval, hours=args.hist_hours, cfg=cfg, verbose=True)
        return

    if args.replay and args.backtest:
        metrics = run_replay_backtest(args.replay, cfg=cfg, verbose=True)
        print("\n=== REPLAY BACKTEST DONE ===")
        for k in ("trades", "wins", "win_rate", "total_pnl", "final_capital", "net_return", "max_drawdown", "return_per_hour"):
            print(f"  {k}: {metrics[k]}")
        return

    optuna_controller = OptunaController(cfg)
    if args.optuna:
        if not OPTUNA_AVAILABLE:
            print("Optuna is not installed.")
            return
        recs = _parse_recordings(args.optuna_recordings)
        ok, info = optuna_controller.start(replay_paths=recs or None, n_trials=args.optuna_trials, train_frac=args.optuna_train_frac, trigger_note="headless CLI")
        if not ok:
            print(f"Optuna could not start: {info}")
            return
        print(f"Optuna started on {info}")
        try:
            while True:
                st = optuna_controller.snapshot()
                print(f"\rstage={st.get('stage')} completed={st.get('completed')}/{st.get('total')} best={st.get('best_score')} msg={st.get('message','')[:80]}", end="", flush=True)
                if not st.get("running"):
                    print()
                    break
                time.sleep(1.0)
        except KeyboardInterrupt:
            optuna_controller.stop()
        return

    if args.replay:
        engine = build_live_engine(cfg, params_source, record_or_replay_paths=[args.replay])
        optuna_controller = OptunaController(cfg, active_record_path_getter=lambda: engine.run_control.active_record_path)
        start_replay_thread(engine, args.replay, args.speed)
        start_capital_tracker_thread(engine)
        if args.no_dashboard:
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                engine.summary_print()
            return
        dash = Dashboard(engine, cfg, params_source=params_source, optuna_controller=optuna_controller, title_suffix=f"  [replay {args.speed}×]")
        try:
            dash.show()
        except KeyboardInterrupt:
            pass
        finally:
            engine.summary_print()
        return

    if not args.record:
        print("Hint: pass --record runs/$(date +%F_%H%M).jsonl to capture this session for replay/backtest/Optuna later.")
    engine = build_live_engine(cfg, params_source)
    optuna_controller = OptunaController(cfg, active_record_path_getter=lambda: engine.run_control.active_record_path)
    auto_watcher = AutoOptunaWatcher(
        engine,
        optuna_controller,
        enabled=args.auto_optuna and OPTUNA_AVAILABLE,
        dd_pct=args.auto_opt_dd_pct,
        idle_min=args.auto_opt_idle_min,
        missed_total=args.auto_opt_missed,
        cooldown_min=args.auto_opt_cooldown_min,
        trials=args.auto_opt_trials,
        train_frac=args.optuna_train_frac,
        fixed_recording=args.record,
    )
    if args.auto_optuna:
        if not OPTUNA_AVAILABLE:
            print("Warning: --auto-optuna ignored because Optuna is not installed.")
        else:
            print(
                f"AUTO_OPTUNA armed: dd≥{args.auto_opt_dd_pct}% OR "
                f"(flat ≥{args.auto_opt_idle_min:g}m & missed≥{args.auto_opt_missed}); "
                f"cooldown {args.auto_opt_cooldown_min:g}m. Params are staged/validated, not hot-applied."
            )
            auto_watcher.start()
    start_ws_thread(engine, record_path=args.record)

    if args.no_dashboard:
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            engine.summary_print()
        return

    dash = Dashboard(engine, cfg, params_source=params_source, optuna_controller=optuna_controller)
    try:
        dash.show()
    except KeyboardInterrupt:
        pass
    finally:
        engine.summary_print()


if __name__ == "__main__":
    main(sys.argv[1:])
