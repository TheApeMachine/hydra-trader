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

Optuna is first-class but experimental. It writes `hydra_candidate.json` after every completed study. It promotes to `hydra_best.json` only when held-out validation beats the baseline config.
