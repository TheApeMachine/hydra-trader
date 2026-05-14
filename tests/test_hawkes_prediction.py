"""Hawkes prediction recording + online learning.

The system pushes a prediction every snapshot, queues it as pending against
the current price, and when the 60s horizon elapses it computes the realised
return and updates the learnable parameters via MSE gradient descent.
"""

import pytest

from hydra.config import Config
from hydra.clock import Clock, VirtualClock
from hydra.market import MarketStore


def _new_store():
    cfg = Config()
    vc = VirtualClock(start=1_000_000.0)
    clock = Clock(vc)
    store = MarketStore(cfg, clock, book_symbols=[], trade_symbols=["BTC/USD"])
    return store, vc, cfg


def _drive_trades(store, vc, symbol, *, n_ticks, buy_bias, drift, base_price=60000.0):
    """Inject synthetic trades + price ticks. Returns the final price."""
    import random
    random.seed(7)
    price = base_price
    for _ in range(n_ticks):
        vc.advance_to(vc.t + 1.0)
        price *= (1.0 + drift + random.gauss(0, 0.0001))
        for _ in range(2):
            side = "buy" if random.random() < buy_bias else "sell"
            qty = random.uniform(0.05, 0.15)
            store.add_trade_to_tape(symbol, side, price, qty)
        # Force a Hawkes snapshot so the prediction step fires
        with store.lock:
            store.hawkes_snapshot_unlocked(symbol, vc.t)
    return price


def test_prediction_is_recorded_each_snapshot():
    store, vc, _cfg = _new_store()
    # Need some trade activity first so excitation/buy/sell are non-zero.
    _drive_trades(store, vc, "BTC/USD", n_ticks=30, buy_bias=0.7, drift=0.0)
    preds = list(store.hawkes_predictions["BTC/USD"])
    assert len(preds) > 0, "predictions must be recorded once trade activity starts"
    # Each entry is (ts_made, predicted_pct, horizon)
    for t, p, h in preds:
        assert h == 60.0
        assert -0.02 <= p <= 0.02, f"prediction must be capped to ±2%, got {p}"


def test_pending_predictions_mature_after_horizon():
    store, vc, _cfg = _new_store()
    # Drive 30 ticks of buying pressure
    _drive_trades(store, vc, "BTC/USD", n_ticks=30, buy_bias=0.7, drift=0.0)
    pending_before = len(store.hawkes_predictions_pending["BTC/USD"])
    n_matured_before = int(store.hawkes_learning["BTC/USD"]["n_matured"])
    # Now advance 90 more seconds with continued activity — all the earlier
    # predictions should mature in that window.
    _drive_trades(store, vc, "BTC/USD", n_ticks=90, buy_bias=0.5, drift=0.0)
    n_matured_after = int(store.hawkes_learning["BTC/USD"]["n_matured"])
    assert n_matured_after >= pending_before, (
        f"matured count should grow past the earlier pending: "
        f"before={pending_before}, n_matured_after={n_matured_after}"
    )
    assert n_matured_after > n_matured_before


def test_pending_predictions_store_original_feature_snapshot():
    store, vc, _cfg = _new_store()
    _drive_trades(store, vc, "BTC/USD", n_ticks=30, buy_bias=0.7, drift=0.0)
    pending = list(store.hawkes_predictions_pending["BTC/USD"])
    assert pending
    assert all(len(item) == 7 for item in pending), (
        "pending Hawkes samples must retain direction/magnitude/slope_norm "
        "from prediction time"
    )


def test_snapshots_do_not_emit_predictions_or_train_without_new_tape():
    store, vc, _cfg = _new_store()
    _drive_trades(store, vc, "BTC/USD", n_ticks=30, buy_bias=0.7, drift=0.0)
    pred_before = len(store.hawkes_predictions["BTC/USD"])
    pending_before = len(store.hawkes_predictions_pending["BTC/USD"])
    matured_before = int(store.hawkes_learning["BTC/USD"]["n_matured"])

    for _ in range(10):
        vc.advance_to(vc.t + 10.0)
        with store.lock:
            store.hawkes_snapshot_unlocked("BTC/USD", vc.t)

    assert len(store.hawkes_predictions["BTC/USD"]) == pred_before
    assert len(store.hawkes_predictions_pending["BTC/USD"]) == pending_before
    assert int(store.hawkes_learning["BTC/USD"]["n_matured"]) == matured_before


def test_learning_uses_stored_features_not_current_flow_state():
    store, vc, _cfg = _new_store()
    sym = "BTC/USD"
    params = store.hawkes_learning[sym]
    params["k"] = 0.001
    params["slope_weight"] = 0.0
    params["bias"] = 0.0
    old_k = float(params["k"])

    t0 = vc.t
    horizon = 60.0
    px0 = 100.0
    direction = 2.0
    magnitude = 1.0
    slope_norm = 0.0
    pred_at_t0 = store._hawkes_prediction_value(params, direction, magnitude, slope_norm)
    store.hawkes_predictions_pending[sym].append(
        (t0, pred_at_t0, px0, horizon, direction, magnitude, slope_norm)
    )

    # Current Hawkes state is deliberately opposite. The delayed update should
    # still increase k because the stored bullish feature underpredicted a +2%
    # realised move.
    store.hawkes_state[sym].update({"buy": 0.0, "sell": 500.0, "last_ts": t0 + horizon})
    store.last_price[sym] = 102.0
    store.price_history[sym].append((t0 + horizon, 102.0))
    store._learn_matured_predictions_unlocked(sym, t0 + horizon)

    assert int(params["n_matured"]) == 1
    assert float(params["last_actual_pct"]) == pytest.approx(0.02)
    assert float(params["k"]) > old_k


def test_learning_status_transitions_cold_to_learning():
    store, vc, _cfg = _new_store()
    # Drive enough ticks to mature ≥50 predictions
    _drive_trades(store, vc, "BTC/USD", n_ticks=60, buy_bias=0.6, drift=0.00005)
    _drive_trades(store, vc, "BTC/USD", n_ticks=120, buy_bias=0.6, drift=0.00005)
    n = int(store.hawkes_learning["BTC/USD"]["n_matured"])
    status = store.hawkes_learning["BTC/USD"]["status"]
    if n < 50:
        assert status == "cold"
    elif n < 200:
        assert status in ("learning", "noisy"), f"got {status} at n={n}"


def test_low_hit_rate_at_high_sample_count_is_not_called_learning():
    """A symbol with 1500 matured predictions at 5% hit rate is BROKEN, not
    'learning'. This is the regression from the live screenshot."""
    store, vc, _cfg = _new_store()
    p = store.hawkes_learning["BTC/USD"]
    p["n_matured"] = 1500
    p["hit_rate"] = 0.05
    p["status"] = "learning"  # what the buggy code produced
    # Apply the status-transition logic by invoking the helper that owns it.
    # We re-invoke the maturation logic with a no-op step so the status
    # field gets re-evaluated.
    store._reclassify_learning_status_unlocked("BTC/USD")
    new_status = store.hawkes_learning["BTC/USD"]["status"]
    assert new_status != "learning", (
        f"1500 samples at 5% hit rate must NOT be 'learning'; got {new_status}"
    )
    assert new_status in ("inverted", "degraded"), (
        f"expected inverted/degraded for systematically wrong predictions, got {new_status}"
    )


def test_high_hit_rate_at_high_sample_count_is_calibrated():
    store, vc, _cfg = _new_store()
    p = store.hawkes_learning["BTC/USD"]
    p["n_matured"] = 500
    p["hit_rate"] = 0.72
    store._reclassify_learning_status_unlocked("BTC/USD")
    assert store.hawkes_learning["BTC/USD"]["status"] == "calibrated"


def test_mid_hit_rate_at_high_sample_count_is_noisy_not_learning():
    """0.45-0.6 hit rate with enough samples means the prediction has no
    real edge but isn't anti-correlated either — call it 'noisy'."""
    store, vc, _cfg = _new_store()
    p = store.hawkes_learning["BTC/USD"]
    p["n_matured"] = 500
    p["hit_rate"] = 0.50
    store._reclassify_learning_status_unlocked("BTC/USD")
    assert store.hawkes_learning["BTC/USD"]["status"] == "noisy"


def test_predictions_are_per_symbol_independent():
    store, vc, _cfg = _new_store()
    # Drive BTC bullish, ETH bearish independently
    import random
    random.seed(11)
    btc_price = 60000.0
    eth_price = 3000.0
    for _ in range(40):
        vc.advance_to(vc.t + 1.0)
        btc_price *= 1.0005  # up
        eth_price *= 0.9995  # down
        for _ in range(2):
            store.add_trade_to_tape("BTC/USD", "buy", btc_price, 0.1)
            store.add_trade_to_tape("ETH/USD", "sell", eth_price, 0.1)
        with store.lock:
            store.hawkes_snapshot_unlocked("BTC/USD", vc.t)
            store.hawkes_snapshot_unlocked("ETH/USD", vc.t)
    btc_preds = list(store.hawkes_predictions["BTC/USD"])
    eth_preds = list(store.hawkes_predictions["ETH/USD"])
    assert btc_preds and eth_preds
    # Median sign of BTC predictions should be positive (buying pressure)
    btc_signs = [1 if p > 0 else -1 if p < 0 else 0 for _, p, _ in btc_preds[-20:]]
    eth_signs = [1 if p > 0 else -1 if p < 0 else 0 for _, p, _ in eth_preds[-20:]]
    assert sum(btc_signs) > 0, f"BTC bullish flow should predict up, got {btc_signs}"
    assert sum(eth_signs) < 0, f"ETH bearish flow should predict down, got {eth_signs}"
