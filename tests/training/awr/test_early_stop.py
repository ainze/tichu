"""Unit tests for the value-baseline early-stop helper.

Pure-Python: no torch imports.
"""

import pytest

from tichu_training.awr.early_stop import EarlyStop


def test_disabled_when_patience_zero():
    """patience<=0 disables the stopper — should never trigger no
    matter how many non-improving chunks are fed."""
    stop = EarlyStop(patience=0, min_delta=0.01)
    for _ in range(1000):
        stop.update(99.0)
    assert not stop.should_stop()


def test_disabled_when_patience_negative():
    stop = EarlyStop(patience=-3, min_delta=0.01)
    for _ in range(1000):
        stop.update(99.0)
    assert not stop.should_stop()


def test_triggers_after_patience_chunks_of_no_improvement():
    """After `patience` consecutive non-improving chunks, latch on."""
    stop = EarlyStop(patience=3, min_delta=0.0)
    stop.update(0.5)   # best=0.5, bad=0
    stop.update(0.5)   # bad=1 (no strict improvement under min_delta=0)
    assert not stop.should_stop()
    stop.update(0.5)   # bad=2
    assert not stop.should_stop()
    stop.update(0.5)   # bad=3 → trigger
    assert stop.should_stop()


def test_counter_resets_on_improvement():
    """An improvement chunk wipes the bad-chunk counter — the next
    `patience` non-improving chunks have to occur consecutively."""
    stop = EarlyStop(patience=3, min_delta=0.0)
    stop.update(0.5)   # best=0.5
    stop.update(0.5)   # bad=1
    stop.update(0.5)   # bad=2
    stop.update(0.4)   # improvement → best=0.4, bad=0
    assert not stop.should_stop()
    stop.update(0.4)   # bad=1
    stop.update(0.4)   # bad=2
    assert not stop.should_stop()
    stop.update(0.4)   # bad=3 → trigger
    assert stop.should_stop()


def test_min_delta_requires_meaningful_improvement():
    """An improvement smaller than min_delta counts as non-improvement.
    Mirrors how baseline_running_mse jitters by tiny amounts even on a
    fully plateaued V — without min_delta, every jitter would reset the
    counter and the stopper would never trigger."""
    stop = EarlyStop(patience=2, min_delta=0.01)
    stop.update(0.500)   # best=0.500
    stop.update(0.499)   # improvement of 0.001 < min_delta → bad=1
    assert not stop.should_stop()
    stop.update(0.498)   # still < min_delta vs best=0.500 → bad=2 → trigger
    assert stop.should_stop()


def test_meaningful_improvement_clears_counter_under_min_delta():
    """A drop of at least min_delta IS an improvement and resets bad."""
    stop = EarlyStop(patience=2, min_delta=0.01)
    stop.update(0.500)
    stop.update(0.499)   # tiny, bad=1
    stop.update(0.480)   # > min_delta vs best=0.500 → improvement, bad=0
    assert not stop.should_stop()
    stop.update(0.480)   # bad=1
    assert not stop.should_stop()
    stop.update(0.480)   # bad=2 → trigger
    assert stop.should_stop()


def test_monotone_decrease_never_triggers():
    """A genuinely-learning baseline (running_mse strictly dropping
    by > min_delta every chunk) should never be early-stopped."""
    stop = EarlyStop(patience=3, min_delta=0.001)
    mse = 0.5
    for _ in range(100):
        mse -= 0.002
        stop.update(mse)
    assert not stop.should_stop()


def test_stop_latches_once_triggered():
    """Once stopped, subsequent improving chunks don't un-stop. Avoids
    accidental restart if the caller forgets to break."""
    stop = EarlyStop(patience=2, min_delta=0.0)
    stop.update(0.5)
    stop.update(0.5)
    stop.update(0.5)
    assert stop.should_stop()
    stop.update(0.1)   # would be an improvement, but stopper is latched
    assert stop.should_stop()


def test_negative_min_delta_rejected():
    """Misconfiguration guard — a negative min_delta would mean
    "improvement counts as non-improvement", which is never sensible."""
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStop(patience=3, min_delta=-0.01)
