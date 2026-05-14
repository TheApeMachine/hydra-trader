"""Readiness score / needle / label must be coherent.

The biggest dashboard UX bug was that the needle drifted around in the WAIT
zone while the label said BUYING SOON — because the needle read raw composite
scores and the label read a separate inertia state with different inputs.

Contract these tests pin down:

1. The needle position for each symbol must be derived from that symbol's
   own Hawkes data (multi_hawkes_history[sym]), not from the focus symbol.
2. The label and the needle must be derived from the SAME underlying score.
   If the label says BUYING SOON, the needle for that symbol must be on the
   buy side of the gauge.
3. A symbol with no Hawkes data must have its needle at zero (WAIT).
"""
import math

import pytest

from hydra.dashboard import Dashboard, _symbol_readiness_score


def _fake_snap(*, focus="BTC/USD", per_symbol_hawkes=None,
               candidates=(), btc_ok=True, btc_flush=False, position=None,
               inertia=None, micro=None):
    """Build a minimal snapshot dict shaped like Engine.dashboard_snapshot."""
    class _View:
        pass
    view = _View()
    view.ts = 1000.0
    view.focus = focus
    view.micro = micro or {}
    view.multi_hawkes_history = per_symbol_hawkes or {}
    view.multi_hawkes_predictions = {}
    view.multi_prices = {}
    view.last_price = 0.0
    view.inertia_state = inertia or {}
    view.btc_backdrop_state = 0
    view.hawkes_learning = {}
    return {
        "focus": focus,
        "market": view,
        "candidates": tuple(candidates),
        "btc_ok": btc_ok,
        "btc_flush": btc_flush,
        "position": position,
        "watch": (),
    }


def test_readiness_score_uses_per_symbol_hawkes_not_focus_only():
    """Each symbol's score must come from its OWN Hawkes history."""
    snap = _fake_snap(
        focus="BTC/USD",
        per_symbol_hawkes={
            "BTC/USD": ((1000.0, 1.0, 1.0, 0.0),),   # neutral
            "ETH/USD": ((1000.0, 3.0, 4.0, 0.5),),   # strong buying excitement
            "SOL/USD": ((1000.0, 3.0, 0.25, -0.5),), # strong selling excitement
        },
    )
    btc_score = _symbol_readiness_score("BTC/USD", snap)
    eth_score = _symbol_readiness_score("ETH/USD", snap)
    sol_score = _symbol_readiness_score("SOL/USD", snap)
    assert -10 <= btc_score <= 10, f"neutral BTC should be near 0, got {btc_score}"
    assert eth_score > 20, f"strong buying ETH should be > 20, got {eth_score}"
    assert sol_score < -20, f"strong selling SOL should be < -20, got {sol_score}"


def test_readiness_score_zero_when_no_hawkes_data():
    """A symbol without history must yield ~0 (modulo BTC backdrop nudge)."""
    snap = _fake_snap(focus="BTC/USD", per_symbol_hawkes={})
    score = _symbol_readiness_score("ETH/USD", snap)
    assert -10 <= score <= 10, f"no data must give near-zero score, got {score}"


def test_btc_flush_nudges_all_symbols_negative():
    snap_calm = _fake_snap(
        focus="BTC/USD",
        per_symbol_hawkes={"ETH/USD": ((1000.0, 2.0, 2.0, 0.0),)},
    )
    snap_flush = _fake_snap(
        focus="BTC/USD",
        per_symbol_hawkes={"ETH/USD": ((1000.0, 2.0, 2.0, 0.0),)},
        btc_flush=True,
    )
    eth_calm = _symbol_readiness_score("ETH/USD", snap_calm)
    eth_flush = _symbol_readiness_score("ETH/USD", snap_flush)
    assert eth_flush < eth_calm - 10, (
        f"BTC flush must penalize all symbols by ≥10; calm={eth_calm:.1f}, flush={eth_flush:.1f}"
    )


def test_score_is_clamped_to_minus100_plus100():
    snap = _fake_snap(
        focus="BTC/USD",
        per_symbol_hawkes={
            "ETH/USD": ((1000.0, 50.0, 1000.0, 5.0),),  # absurd readings
        },
        candidates=(
            {"symbol": "ETH/USD", "score": 100.0, "regime": "macro_thrust"},
        ),
    )
    s = _symbol_readiness_score("ETH/USD", snap)
    assert -100.0 <= s <= 100.0


def test_label_and_needle_use_same_score():
    """The label state must be a function of the same score as the needle.

    This is the regression check: in the buggy version the needle position
    used raw composite (per-symbol) but the label used an unrelated inertia
    state from market.inertia_state. They could disagree.
    """
    from hydra.dashboard import _label_for_score
    # Score above the BUYING_SOON threshold must yield BUYING SOON label.
    assert _label_for_score(70.0) == ("BUYING SOON", "positive")
    assert _label_for_score(30.0) == ("ALMOST", "neutral")
    assert _label_for_score(0.0) == ("WAITING", "neutral")
    assert _label_for_score(-50.0) == ("STAY OUT", "negative")
    # Boundary: just below 60 is ALMOST, at/above 60 is BUYING SOON.
    assert _label_for_score(59.9)[0] == "ALMOST"
    assert _label_for_score(60.0)[0] == "BUYING SOON"
