# Hydra v5 modular paper trader

This is the v5 modular split of the Hydra Kraken paper-trading lab.

Important distinction:

- `--historical-backtest` is a simple Kraken REST OHLC candle-only test. It cannot test trade tape, L2 book imbalance, `book_ignition`, or L2 paper fills.
- `--replay FILE --backtest` is the full-signal backtest from a recorded websocket stream. This is the correct input for Optuna.

Recommended workflow:

```bash
python master.py --record runs/session1.jsonl --no-params
python master.py --replay runs/session1.jsonl --backtest --no-params
python master.py --optuna --optuna-recordings runs/session1.jsonl --optuna-trials 50 --no-params
python master.py
```

Optuna is first-class but experimental. It writes `hydra_candidate.json` after every completed study. It promotes to `hydra_best.json` only when held-out validation beats the baseline config on both score and validation return. At the end of a study it validates a small shortlist of the best train trials, so a lower train-score candidate can still be promoted if it generalizes better.

Built-in defaults keep `macro_thrust` disabled because the bundled recordings show that ungated thrust churn dominates losses. Tuned JSON can re-enable it with `macro_thrust_enabled: true`; legacy tuned files that already contain thrust/`mt_*` parameters are treated as thrust-enabled for compatibility.

Micro `book_ignition` now includes a lightweight online Hawkes-style tape filter: trade notional impulses are clipped, square-root weighted, exponentially decayed, and compared with a rolling baseline to require live buy-side self-excitation before entry. The same state exposes an optional sell-excitation flip exit for Optuna/live experimentation.

The paper wallet also maintains explicit simulated stop-market orders for open long positions. Initial hard stops are installed on entry, trailing stops ratchet upward without lowering, and OHLC-only backtests trigger stops on candle lows so paper behavior is closer to a real resting stop order.
