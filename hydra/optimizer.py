from __future__ import annotations

import dataclasses
import json
import math
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .backtest import run_replay_backtest
from .config import CANDIDATE_PARAMS_PATH, DEFAULT_PARAMS_PATH, Config, apply_params
from .data import filter_paths_for_shared_read, list_recordings, replay_messages

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    optuna = None
    OPTUNA_AVAILABLE = False


class OptunaController:
    """Threaded Optuna runner that never touches live strategy state."""

    def __init__(self, base_cfg: Config, active_record_path_getter=lambda: None):
        self.base_cfg = dataclasses.replace(base_cfg)
        self.active_record_path_getter = active_record_path_getter
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = {
            "running": False,
            "stage": "idle",
            "trials": [],
            "best_score": None,
            "best_trial": None,
            "best_params": None,
            "completed": 0,
            "total": 0,
            "started_at": None,
            "elapsed": 0.0,
            "message": "",
            "last_error": "",
            "replay_path": None,
            "replay_paths": [],
            "trigger_note": "",
            "should_stop": False,
            "current_trial": None,
            "current_file": "",
            "candidate_path": str(CANDIDATE_PARAMS_PATH),
            "best_path": str(DEFAULT_PARAMS_PATH),
            "promoted": False,
            "baseline_validation_aggregate": None,
            "candidate_validation_aggregate": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            out = dict(self.state)
            out["trials"] = list(self.state.get("trials", []))
            return out

    def start(
        self,
        replay_path: str | None = None,
        replay_paths: list[str] | None = None,
        n_trials: int = 100,
        train_frac: float = 0.7,
        seed: int = 42,
        trigger_note: str | None = None,
        robust_aggregate: bool = True,
    ) -> tuple[bool, str]:
        if not OPTUNA_AVAILABLE:
            return False, "optuna not installed; pip install optuna"
        with self.lock:
            if self.state["running"]:
                return False, "already running"

        if replay_paths:
            paths = [p for p in replay_paths if os.path.exists(p)]
        elif replay_path:
            paths = [replay_path] if os.path.exists(replay_path) else []
        else:
            paths = list_recordings(limit=5)
        paths = filter_paths_for_shared_read(paths, active_record_path=self.active_record_path_getter(), min_records=5)
        if not paths:
            return False, "no usable closed recordings found"

        display_path = paths[0] if len(paths) == 1 else f"{len(paths)} recordings (walk-forward)"
        with self.lock:
            self.state.update({
                "running": True,
                "stage": "starting",
                "trials": [],
                "best_score": None,
                "best_trial": None,
                "best_params": None,
                "completed": 0,
                "total": int(n_trials),
                "started_at": time.time(),
                "elapsed": 0.0,
                "message": f"running on {display_path}",
                "last_error": "",
                "replay_path": display_path,
                "replay_paths": list(paths),
                "trigger_note": trigger_note or "",
                "should_stop": False,
                "current_trial": None,
                "current_file": "",
                "promoted": False,
                "baseline_validation_aggregate": None,
                "candidate_validation_aggregate": None,
            })
        self.thread = threading.Thread(
            target=self._run_thread,
            args=(paths, n_trials, train_frac, seed, robust_aggregate),
            name="hydra-optuna",
            daemon=True,
        )
        self.thread.start()
        return True, display_path

    def stop(self) -> None:
        with self.lock:
            if self.state["running"]:
                self.state["should_stop"] = True
                self.state["message"] = "stopping after current trial…"

    def _set(self, **kwargs) -> None:
        with self.lock:
            self.state.update(kwargs)

    def _get_should_stop(self) -> bool:
        with self.lock:
            return bool(self.state.get("should_stop"))

    @staticmethod
    def split_recording(path: str, train_frac: float) -> tuple[float, float, float]:
        timestamps = [t for t, _ in replay_messages(path, allow_partial=True)]
        if len(timestamps) < 10:
            raise RuntimeError(f"recording {path} has too few readable messages ({len(timestamps)})")
        cutoff_idx = max(1, min(len(timestamps) - 1, int(len(timestamps) * train_frac)))
        return timestamps[0], timestamps[cutoff_idx], timestamps[-1]

    @staticmethod
    def _score(metrics: dict[str, Any], cfg: Config) -> float:
        """Higher is better. Inactive runs must not collapse to a single constant
        (previously everything with trades<2 scored -8.0 so trial 0 always 'won')."""
        trades = int(metrics.get("trades", 0) or 0)
        ret = float(metrics.get("net_return", 0.0) or 0.0)
        msgs = float(metrics.get("messages", 0) or 0.0)

        if trades == 0:
            # Strong penalty; tiny spread so Optuna still prefers runs that at least move Capital/equity
            return -1.0e6 + min(msgs / 250_000.0, 80.0)
        if trades == 1:
            return (
                -50_000.0
                + 2_500.0 * float(metrics.get("win_rate", 0.0) or 0.0)
                + 15_000.0 * ret
                + min(msgs / 200_000.0, 40.0)
            )

        rph = float(metrics.get("return_per_hour", 0.0) or 0.0)
        dd = float(metrics.get("max_drawdown", 0.0) or 0.0)
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        worst_trade_pct = abs(float(metrics.get("worst_trade", 0.0) or 0.0)) / max(cfg.start_capital, 1e-9)
        avg_hold_min = float(metrics.get("avg_hold_sec", 0.0) or 0.0) / 60.0
        activity = 12.0 * math.log1p(float(trades))
        return (
            900.0 * rph
            + 350.0 * ret
            + 40.0 * win_rate
            - 1800.0 * dd
            - 350.0 * worst_trade_pct
            - 1.5 * avg_hold_min
            + activity
        )

    @staticmethod
    def _aggregate(scores: list[float], robust: bool) -> float:
        if not scores:
            return -9999.0
        mean_score = sum(scores) / len(scores)
        if robust and len(scores) > 1:
            return 0.6 * min(scores) + 0.4 * mean_score
        return mean_score

    def _run_segment(self, replay_path: str, cfg: Config, t_start: float, t_end: float, lower_inclusive: bool = True, symbol_sets=None) -> dict[str, Any]:
        from .clock import Clock, VirtualClock
        from .engine import HydraEngine
        from .data import infer_recording_symbols

        if symbol_sets is None:
            symbols, book_syms, trade_syms = infer_recording_symbols([replay_path])
        else:
            symbols, book_syms, trade_syms = symbol_sets
        clock = Clock()
        vclock = VirtualClock()
        clock.set(vclock)
        engine = HydraEngine(cfg, symbols=symbols, book_symbols=book_syms, trade_symbols=trade_syms, clock=clock, name="HydraOptuna", csv_enabled=False, verbose=False)
        last_cap_t = 0.0
        n = 0
        for t, m in replay_messages(replay_path, allow_partial=True):
            if lower_inclusive:
                if t < t_start:
                    continue
            else:
                if t <= t_start:
                    continue
            if t > t_end:
                break
            vclock.advance_to(t)
            if m is None:
                continue
            engine._process_message_unguarded(m)
            n += 1
            if t - last_cap_t >= 1.0:
                engine.sample_capital()
                last_cap_t = t
        engine.force_exit_eob()
        metrics = engine.metrics()
        metrics["messages"] = n
        return metrics

    def _sample_config(self, trial, base: Config) -> Config:
        cfg = dataclasses.replace(base)
        cfg.micro_min_trades = trial.suggest_int("micro_min_trades", 5, 20)
        cfg.micro_min_buy_notional = trial.suggest_float("micro_min_buy_notional", 2_000, 50_000, log=True)
        cfg.micro_burst_multiple = trial.suggest_float("micro_burst_multiple", 2.0, 8.0)
        cfg.micro_min_imbalance = trial.suggest_float("micro_min_imbalance", 1.2, 5.0)
        cfg.micro_min_move_pct = trial.suggest_float("micro_min_move_pct", 0.003, 0.015)
        cfg.micro_cooldown_sec = trial.suggest_float("micro_cooldown_sec", 30.0, 240.0)
        cfg.micro_candidate_cooldown_sec = trial.suggest_float("micro_candidate_cooldown_sec", 2.0, 8.0)
        cfg.micro_accel_threshold = trial.suggest_float("micro_accel_threshold", 1.0, 3.5)
        cfg.micro_delta_divergence = trial.suggest_float("micro_delta_divergence", 0.7, 2.8)

        cfg.macro_min_tape_trades = trial.suggest_int("macro_min_tape_trades", 3, 15)
        cfg.macro_min_tape_notional = trial.suggest_float("macro_min_tape_notional", 1_000, 30_000, log=True)
        cfg.macro_min_tape_buy_share = trial.suggest_float("macro_min_tape_buy_share", 0.50, 0.68)
        cfg.macro_max_spread_bps = trial.suggest_float("macro_max_spread_bps", 10.0, 150.0)
        cfg.macro_min_book_imbalance = trial.suggest_float("macro_min_book_imbalance", 0.70, 1.30)
        cfg.macro_max_vol_x = trial.suggest_float("macro_max_vol_x", 12.0, 120.0)
        cfg.macro_require_microstructure = trial.suggest_categorical("macro_require_microstructure", [False, True])

        cfg.max_portfolio_heat = trial.suggest_float("max_portfolio_heat", 0.35, 0.80)
        cfg.daily_loss_limit = trial.suggest_float("daily_loss_limit", 0.05, 0.22)
        cfg.vol_target_atr = trial.suggest_float("vol_target_atr", 0.005, 0.024)
        cfg.liq_buffer = trial.suggest_float("liq_buffer", 45_000, 450_000, log=True)

        cfg.thrust_min_ret_3m = trial.suggest_float("thrust_min_ret_3m", 0.005, 0.025)
        cfg.thrust_min_ret_5m = trial.suggest_float("thrust_min_ret_5m", 0.010, 0.040)
        cfg.thrust_min_vol_x = trial.suggest_float("thrust_min_vol_x", 1.5, 6.0)
        cfg.thrust_min_close_pos = trial.suggest_float("thrust_min_close_pos", 0.55, 0.92)

        cfg.bi_hard_stop = trial.suggest_float("bi_hard_stop", 0.010, 0.060)
        cfg.bi_trail_activate = trial.suggest_float("bi_trail_activate", 0.010, 0.060)
        cfg.bi_trail_pct = trial.suggest_float("bi_trail_pct", 0.005, 0.040)
        cfg.bi_take_profit = trial.suggest_float("bi_take_profit", 0.010, 0.045)
        cfg.bi_max_hold = trial.suggest_float("bi_max_hold", 300.0, 1800.0)
        cfg.bi_stall_check = trial.suggest_float("bi_stall_check", 60.0, 600.0)
        cfg.bi_stall_ret = trial.suggest_float("bi_stall_ret", 0.001, 0.020)

        cfg.mt_hard_stop = trial.suggest_float("mt_hard_stop", 0.015, 0.070)
        cfg.mt_trail_activate = trial.suggest_float("mt_trail_activate", 0.012, 0.060)
        cfg.mt_trail_pct = trial.suggest_float("mt_trail_pct", 0.006, 0.040)
        cfg.mt_take_profit = trial.suggest_float("mt_take_profit", 0.014, 0.070)
        cfg.mt_max_hold = trial.suggest_float("mt_max_hold", 600.0, 5400.0)
        cfg.mt_stall_check = trial.suggest_float("mt_stall_check", 180.0, 1800.0)
        cfg.mt_stall_ret = trial.suggest_float("mt_stall_ret", 0.001, 0.020)

        cfg.mr_hard_stop = trial.suggest_float("mr_hard_stop", 0.020, 0.070)
        cfg.mr_trail_activate = trial.suggest_float("mr_trail_activate", 0.014, 0.060)
        cfg.mr_trail_pct = trial.suggest_float("mr_trail_pct", 0.006, 0.040)
        cfg.mr_take_profit = trial.suggest_float("mr_take_profit", 0.014, 0.075)
        cfg.mr_max_hold = trial.suggest_float("mr_max_hold", 900.0, 7200.0)
        cfg.mr_stall_check = trial.suggest_float("mr_stall_check", 180.0, 1800.0)
        cfg.mr_stall_ret = trial.suggest_float("mr_stall_ret", 0.001, 0.020)

        cfg.macro_early_fail_sec = trial.suggest_float("macro_early_fail_sec", 90.0, 420.0)
        cfg.macro_early_fail_ret = trial.suggest_float("macro_early_fail_ret", -0.025, -0.004)
        cfg.macro_no_followthrough_sec = trial.suggest_float("macro_no_followthrough_sec", 180.0, 900.0)
        cfg.macro_no_followthrough_ret = trial.suggest_float("macro_no_followthrough_ret", -0.002, 0.012)
        return cfg

    def _run_thread(self, replay_paths: list[str], n_trials: int, train_frac: float, seed: int, robust_aggregate: bool) -> None:
        t0 = time.time()
        try:
            self._set(stage="splitting", message="splitting recordings…")
            print("[OPTUNA] cwd:", os.getcwd())
            print("[OPTUNA] candidate output:", CANDIDATE_PARAMS_PATH.resolve())
            print("[OPTUNA] best output:", DEFAULT_PARAMS_PATH.resolve())
            print("[OPTUNA] recordings:")
            splits = []
            symbol_cache = {}
            from .data import infer_recording_symbols
            for path in replay_paths:
                print("  ", os.path.abspath(path), os.path.getsize(path), "bytes")
                t_start, t_split, t_end = self.split_recording(path, train_frac)
                splits.append((path, t_start, t_split, t_end))
                self._set(stage="indexing", current_file=os.path.basename(path), message="indexing symbols…")
                symbol_cache[path] = infer_recording_symbols([path])

            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            base_cfg = dataclasses.replace(self.base_cfg)

            def objective(trial):
                cfg = self._sample_config(trial, base_cfg)
                scores = []
                agg_trades = agg_wins = 0
                agg_max_dd = 0.0
                agg_final_total = 0.0
                net_returns = []
                self._set(stage="trial", current_trial=trial.number)
                for path, t_start, t_split, _t_end in splits:
                    self._set(current_file=os.path.basename(path))
                    m = self._run_segment(path, cfg, t_start, t_split, symbol_sets=symbol_cache.get(path))
                    s = self._score(m, cfg)
                    scores.append(s)
                    agg_trades += m["trades"]
                    agg_wins += m["wins"]
                    agg_max_dd = max(agg_max_dd, m["max_drawdown"])
                    agg_final_total += m["final_capital"]
                    net_returns.append(m["net_return"])
                mean_score = sum(scores) / len(scores)
                min_score = min(scores)
                win_rate = (agg_wins / agg_trades) if agg_trades else 0.0
                trial.set_user_attr("trades", agg_trades)
                trial.set_user_attr("win_rate", win_rate)
                trial.set_user_attr("net_return", sum(net_returns) / len(net_returns) if net_returns else 0.0)
                trial.set_user_attr("max_drawdown", agg_max_dd)
                trial.set_user_attr("final_capital", agg_final_total / len(splits) if splits else 0.0)
                trial.set_user_attr("scores_per_recording", scores)
                trial.set_user_attr("min_score", min_score)
                trial.set_user_attr("mean_score", mean_score)
                trial.set_user_attr("n_recordings", len(splits))
                return self._aggregate(scores, robust_aggregate)

            def callback(study, trial):
                if trial.state != optuna.trial.TrialState.COMPLETE:
                    return
                try:
                    best_val = study.best_value
                    best_num = study.best_trial.number
                except ValueError:
                    best_val = trial.value
                    best_num = trial.number
                is_best = trial.number == best_num
                elapsed = time.time() - t0
                with self.lock:
                    self.state["trials"].append({
                        "n": trial.number,
                        "score": trial.value,
                        "best_score": best_val,
                        "is_best": is_best,
                        "attrs": dict(trial.user_attrs),
                        "params": dict(trial.params),
                        "elapsed": elapsed,
                    })
                    self.state["trials"] = self.state["trials"][-80:]
                    self.state["completed"] = trial.number + 1
                    self.state["elapsed"] = elapsed
                    self.state["stage"] = "trial"
                    if is_best:
                        self.state["best_score"] = trial.value
                        self.state["best_trial"] = trial.number
                        self.state["best_params"] = dict(trial.params)
                    if self.state.get("should_stop"):
                        study.stop()

            self._set(stage="optimizing")
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])
            try:
                best_trial = study.best_trial
            except ValueError:
                self._set(stage="done", message="done: no complete trials")
                return

            best_cfg = apply_params(dataclasses.replace(base_cfg), dict(best_trial.params))
            self._set(stage="validating", message="validating best params…")
            val_scores = []
            val_rows = []
            base_val_scores = []
            for path, _t0, t_split, t_end in splits:
                mv = self._run_segment(path, best_cfg, t_split, t_end, lower_inclusive=False, symbol_sets=symbol_cache.get(path))
                sv = self._score(mv, best_cfg)
                val_scores.append(sv)
                val_rows.append({"file": os.path.basename(path), "score": sv, "metrics": mv})
                mb = self._run_segment(path, base_cfg, t_split, t_end, lower_inclusive=False, symbol_sets=symbol_cache.get(path))
                base_val_scores.append(self._score(mb, base_cfg))

            cand_val_agg = self._aggregate(val_scores, robust_aggregate)
            base_val_agg = self._aggregate(base_val_scores, robust_aggregate)

            train_trades = int(best_trial.user_attrs.get("trades", 0) or 0)
            val_trade_sum = sum(int(r["metrics"].get("trades", 0) or 0) for r in val_rows)
            has_activity = train_trades >= 2 and val_trade_sum >= 1
            beats_baseline = cand_val_agg > base_val_agg
            promoted = bool(beats_baseline and has_activity)

            out = {
                "study_name": "hydra_v5_independent",
                "best_score": study.best_value,
                "best_params": dict(best_trial.params),
                "best_train_attrs": dict(best_trial.user_attrs),
                "best_validation_scores": val_scores,
                "best_validation_by_recording": val_rows,
                "best_validation_aggregate": cand_val_agg,
                "baseline_validation_scores": base_val_scores,
                "baseline_validation_aggregate": base_val_agg,
                "promoted_to_best": promoted,
                "promotion_gate": {
                    "train_trades": train_trades,
                    "val_trade_sum": val_trade_sum,
                    "beats_baseline": beats_baseline,
                    "has_min_activity": has_activity,
                },
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "replay_paths": list(replay_paths),
                "robust_aggregate": robust_aggregate,
            }
            CANDIDATE_PARAMS_PATH.write_text(json.dumps(out, indent=2, default=str))
            if promoted:
                DEFAULT_PARAMS_PATH.write_text(json.dumps(out, indent=2, default=str))
                msg = f"done — candidate beat baseline ({cand_val_agg:.4f} > {base_val_agg:.4f}) with activity; wrote {DEFAULT_PARAMS_PATH.resolve()}"
            else:
                reason = []
                if not beats_baseline:
                    reason.append(f"score {cand_val_agg:.4f} ≤ baseline {base_val_agg:.4f}")
                if not has_activity:
                    reason.append(f"insufficient trades (train={train_trades}, val_sum={val_trade_sum})")
                msg = f"done — NOT promoted ({'; '.join(reason)}); wrote {CANDIDATE_PARAMS_PATH.resolve()} only"
            print("[OPTUNA]", msg)
            self._set(
                stage="done",
                message=msg,
                promoted=promoted,
                candidate_validation_aggregate=cand_val_agg,
                baseline_validation_aggregate=base_val_agg,
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print("[OPTUNA] error:", err)
            traceback.print_exc()
            self._set(stage="error", message=f"error: {err}", last_error=err)
        finally:
            with self.lock:
                self.state["running"] = False
                self.state["should_stop"] = False
                self.state["elapsed"] = time.time() - t0


class AutoOptunaWatcher:
    """Experimental but first-class auto tuning trigger.

    It starts Optuna from closed recordings when drawdown or idle+missed
    thresholds fire. It does not apply params live; Optuna writes candidate/best
    files based on validation.
    """

    def __init__(
        self,
        engine,
        controller: OptunaController,
        enabled: bool = False,
        dd_pct: float = 5.0,
        idle_min: float = 45.0,
        missed_total: int = 8,
        cooldown_min: float = 60.0,
        trials: int = 100,
        train_frac: float = 0.7,
        fixed_recording: str | None = None,
    ):
        self.engine = engine
        self.controller = controller
        self.enabled = enabled
        self.dd_pct = dd_pct
        self.idle_min = idle_min
        self.missed_total = missed_total
        self.cooldown_min = cooldown_min
        self.trials = trials
        self.train_frac = train_frac
        self.fixed_recording = fixed_recording
        self.last_completed_wall = 0.0
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        if not self.enabled or not OPTUNA_AVAILABLE:
            return
        self.engine.auto_stats.boot()
        self.thread = threading.Thread(target=self._loop, name="hydra-auto-optuna", daemon=True)
        self.thread.start()

    def _resolve_recordings(self) -> list[str]:
        candidates = []
        if self.fixed_recording and os.path.isfile(self.fixed_recording):
            candidates.append(self.fixed_recording)
        candidates.extend(list_recordings(limit=5))
        seen, unique = set(), []
        for p in candidates:
            ap = os.path.normpath(os.path.abspath(p))
            if ap not in seen:
                seen.add(ap)
                unique.append(p)
        return filter_paths_for_shared_read(unique, active_record_path=self.engine.run_control.active_record_path, min_records=5)[:3]

    def _loop(self) -> None:
        last_no_rec_log = 0.0
        while not self.stop_event.is_set():
            time.sleep(15.0)
            if not self.enabled:
                continue
            opt = self.controller.snapshot()
            if opt.get("running"):
                continue
            if self.last_completed_wall and (time.time() - self.last_completed_wall) < self.cooldown_min * 60.0:
                continue
            paths = self._resolve_recordings()
            if not paths:
                if time.time() - last_no_rec_log > 300.0:
                    print("[AUTO_OPTUNA] no usable closed recording found yet")
                    last_no_rec_log = time.time()
                continue
            equity = self.engine.capital()
            snap = self.engine.auto_stats.snapshot(equity)
            dd_hit = snap["dd_pct"] >= self.dd_pct
            inact_hit = (not self.engine.trader.pos and snap["idle_min"] >= self.idle_min and snap["missed"] >= self.missed_total)
            if not (dd_hit or inact_hit):
                continue
            if self.engine.trader.pos:
                continue
            parts = []
            if dd_hit:
                parts.append(f"drawdown {snap['dd_pct']:.2f}% ≥ {self.dd_pct}%")
            if inact_hit:
                parts.append(f"flat {snap['idle_min']:.1f}m & missed {snap['missed']} ≥ {self.missed_total}")
            note = "; ".join(parts)
            ok, info = self.controller.start(replay_paths=paths, n_trials=self.trials, train_frac=self.train_frac, trigger_note=note)
            if ok:
                self.last_completed_wall = time.time()
                print(f"[AUTO_OPTUNA] started on {info} ({note})")
            else:
                print(f"[AUTO_OPTUNA] could not start: {info}")
