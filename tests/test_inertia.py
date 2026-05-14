"""InertiaTracker: sticky direction with hysteresis.

State ∈ {-1, 0, +1}. Flipping requires the raw value to cross the relevant
threshold AND stay across it for ``confirm_ticks`` consecutive updates.
"""
from hydra.utils import InertiaTracker


def test_initial_state_is_zero():
    tr = InertiaTracker(confirm_ticks=3)
    assert tr.state == 0


def test_single_tick_above_threshold_does_not_flip():
    tr = InertiaTracker(confirm_ticks=3)
    tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 0, "one tick should not be enough to flip"


def test_sustained_ticks_flip_state():
    tr = InertiaTracker(confirm_ticks=3)
    for _ in range(3):
        tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1


def test_oscillating_signal_does_not_flip():
    """If raw signal flickers above/below threshold, state must NOT flip."""
    tr = InertiaTracker(confirm_ticks=3)
    for i in range(20):
        v = 1.0 if i % 2 == 0 else -1.0
        tr.update(v, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 0, "oscillation should leave state at 0"


def test_flip_requires_sustained_opposite_signal():
    tr = InertiaTracker(confirm_ticks=3)
    for _ in range(3):
        tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
    # One opposite tick must not flip
    tr.update(-1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
    tr.update(-1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
    # Third sustained opposite tick flips
    tr.update(-1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == -1


def test_returning_to_neutral_zone_requires_sustained_evidence():
    tr = InertiaTracker(confirm_ticks=3)
    for _ in range(3):
        tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
    # Single neutral reading must not reset
    tr.update(0.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
    # Three sustained neutral readings → return to 0
    tr.update(0.0, pos_threshold=0.5, neg_threshold=-0.5)
    tr.update(0.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 0


def test_reset_clears_state_and_pending():
    tr = InertiaTracker(confirm_ticks=3)
    for _ in range(3):
        tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
    tr.reset()
    assert tr.state == 0
    # After reset, a single tick still shouldn't flip
    tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 0


def test_asymmetric_thresholds():
    """A flush is easier than a recovery — asymmetric thresholds must work."""
    tr = InertiaTracker(confirm_ticks=2)
    # Negative side at -0.01 is sticky-bearish
    for _ in range(2):
        tr.update(-0.02, pos_threshold=0.05, neg_threshold=-0.01)
    assert tr.state == -1
    # A small positive bounce shouldn't flip back
    tr.update(0.02, pos_threshold=0.05, neg_threshold=-0.01)
    assert tr.state == -1
    # Need sustained ≥0.05 to flip
    tr.update(0.06, pos_threshold=0.05, neg_threshold=-0.01)
    tr.update(0.06, pos_threshold=0.05, neg_threshold=-0.01)
    assert tr.state == 1


def test_confirm_ticks_of_one_means_instant_flip():
    tr = InertiaTracker(confirm_ticks=1)
    tr.update(1.0, pos_threshold=0.5, neg_threshold=-0.5)
    assert tr.state == 1
