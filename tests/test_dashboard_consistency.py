"""Dashboard cross-panel consistency.

These tests pin the contracts that the live screenshots violated:

- Aggregate Hawkes hit count in the title MUST equal the sum of per-symbol
  hit counts shown in the legend. (Bug: they used different windows.)
- Market-overview title verdict and the caption's per-symbol up/down/flat
  tags MUST agree on what counts as "flat". (Bug: title used 0.1%, caption
  used 0.3%.)
"""
from hydra.dashboard import (
    _hawkes_hit_aggregate,
    _market_overview_verdict,
    _per_symbol_change_label,
    _MARKET_FLAT_THRESHOLD_PCT,
)


# ── hit-count aggregation ────────────────────────────────────────────
def test_aggregate_hits_equal_sum_of_per_symbol():
    per_symbol = {"BTC/USD": (16, 16), "ETH/USD": (6, 8), "SOL/USD": (6, 6)}
    total_agree, total_n = _hawkes_hit_aggregate(per_symbol)
    assert total_agree == 28
    assert total_n == 30


def test_aggregate_with_empty_inputs():
    assert _hawkes_hit_aggregate({}) == (0, 0)
    assert _hawkes_hit_aggregate({"BTC/USD": (0, 0)}) == (0, 0)


def test_aggregate_handles_partial_symbols():
    per_symbol = {"BTC/USD": (10, 10), "ETH/USD": (0, 0)}
    assert _hawkes_hit_aggregate(per_symbol) == (10, 10)


# ── market overview consistency ──────────────────────────────────────
def test_market_overview_threshold_is_single_value():
    """The flat-threshold constant must exist and be the value both title
    and caption use."""
    assert _MARKET_FLAT_THRESHOLD_PCT > 0
    # Anything below the threshold (in absolute value) is "flat"
    assert _per_symbol_change_label(_MARKET_FLAT_THRESHOLD_PCT * 0.5) == "flat"
    # At or above the threshold, it's directional
    assert _per_symbol_change_label(_MARKET_FLAT_THRESHOLD_PCT + 0.01) == "up"
    assert _per_symbol_change_label(-(_MARKET_FLAT_THRESHOLD_PCT + 0.01)) == "down"


def test_caption_and_title_agree_on_flat_symbol():
    """If a coin is labeled 'flat' in the caption, the title must not count
    it toward 'broad strength' or 'broad weakness'."""
    # All three flat
    changes = {"BTC/USD": 0.05, "ETH/USD": 0.0, "SOL/USD": -0.04}
    mood, verdict = _market_overview_verdict(changes)
    assert mood == "mixed"
    assert verdict == "neutral"
    # All three up
    changes = {"BTC/USD": 0.5, "ETH/USD": 0.6, "SOL/USD": 0.4}
    mood, verdict = _market_overview_verdict(changes)
    assert mood == "broad strength"
    assert verdict == "positive"
    # Mixed: BTC up, ETH flat, SOL down
    changes = {"BTC/USD": 0.5, "ETH/USD": 0.0, "SOL/USD": -0.5}
    mood, verdict = _market_overview_verdict(changes)
    assert mood == "mixed"
    assert verdict == "neutral"


def test_no_broad_strength_when_only_one_symbol_up():
    """A single coin up does not count as 'broad' strength."""
    changes = {"BTC/USD": 0.5, "ETH/USD": 0.0, "SOL/USD": 0.0}
    mood, verdict = _market_overview_verdict(changes)
    assert "broad" not in mood
    assert verdict != "positive"
