"""PromotionGate — promote only on a CI-validated margin against ALL opponents."""

import numpy as np
import pytest

from tichu_training.ppo.promotion_gate import PromotionGate


def _normal(mean, sd, n, seed=0):
    return list(np.random.default_rng(seed).normal(mean, sd, n))


def _champ(v):
    return v["opponents"]["champion"]


def test_not_ready_before_window_full():
    gate = PromotionGate(window_games=1000)
    gate.record("champion", _normal(50, 100, 400))
    assert not gate.ready()
    assert gate.verdict()["promote"] is False


def test_strongly_positive_margin_promotes():
    gate = PromotionGate(window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))  # +40/game => CI well > 0
    v = gate.verdict()
    assert _champ(v)["n"] == 2000
    assert _champ(v)["ci_lo"] > 0.0
    assert v["promote"] is True


def test_margin_straddling_zero_does_not_promote():
    gate = PromotionGate(window_games=2000)
    gate.record("champion", _normal(0.0, 100, 2000))
    v = gate.verdict()
    assert _champ(v)["ci_lo"] < 0.0 < _champ(v)["ci_hi"]
    assert v["promote"] is False


def test_threshold_must_be_cleared_not_just_zero():
    gate = PromotionGate(window_games=4000, threshold=20.0)
    gate.record("champion", _normal(15.0, 60, 4000))  # CI ~[13,17]
    assert gate.verdict()["promote"] is False
    gate0 = PromotionGate(window_games=4000, threshold=0.0)
    gate0.record("champion", _normal(15.0, 60, 4000))
    assert gate0.verdict()["promote"] is True


def test_record_accumulates_across_iters_and_reset_clears():
    gate = PromotionGate(window_games=1000)
    for _ in range(5):
        gate.record("champion", _normal(30, 100, 256))  # 5 x 256 = 1280
    assert gate.ready() and gate.n("champion") == 1280
    gate.reset()
    assert gate.n("champion") == 0 and not gate.ready()


def test_window_games_must_be_positive():
    with pytest.raises(ValueError):
        PromotionGate(window_games=0)


def test_unknown_opponent_rejected():
    gate = PromotionGate(window_games=10)
    with pytest.raises(KeyError):
        gate.record("bc", [1.0])


# --- multi-opponent: must beat EVERY opponent (the anti-cycling / beat-both gate) ---

def test_multi_opponent_requires_all_to_clear():
    gate = PromotionGate(opponents=("champion", "bc"), window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))  # clears
    gate.record("bc", _normal(0.0, 100, 2000))        # straddles 0
    v = gate.verdict()
    assert _champ(v)["ci_lo"] > 0.0
    assert v["opponents"]["bc"]["ci_lo"] < 0.0
    assert v["promote"] is False  # bc fails => no promotion despite beating champion


def test_multi_opponent_not_ready_until_all_windows_full():
    gate = PromotionGate(opponents=("champion", "bc"), window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))   # champion full
    # bc not recorded yet
    assert not gate.ready()
    assert gate.verdict()["promote"] is False
    gate.record("bc", _normal(40, 100, 2000))         # now both full + positive
    assert gate.ready()
    assert gate.verdict()["promote"] is True


# --- observe-only opponents: scored + reported every window, NEVER required ---------
# (e.g. the shipped champion as a fixed visibility reference while the promote
# decision stays {champion, bc} — plot panel 6 picks the stream up from the CSV.)

def test_observe_only_losing_stream_does_not_block_promotion():
    gate = PromotionGate(opponents=("champion", "bc", "watch"),
                         observe_only=("watch",), window_games=2000)
    gate.record("champion", _normal(40, 100, 2000))   # clears
    gate.record("bc", _normal(40, 100, 2000, seed=1))  # clears
    gate.record("watch", _normal(-40, 100, 2000, seed=2))  # would block if required
    v = gate.verdict()
    assert v["opponents"]["watch"]["ci_hi"] < 0.0  # decisively losing
    assert v["promote"] is True                     # ...and irrelevant to the verdict


def test_observe_only_stream_is_still_reported_in_the_verdict():
    gate = PromotionGate(opponents=("champion", "watch"),
                         observe_only=("watch",), window_games=1000)
    gate.record("champion", _normal(40, 100, 1000))
    gate.record("watch", _normal(-10, 100, 500))
    v = gate.verdict()
    assert v["opponents"]["watch"]["n"] == 500  # reported (the CSV/plot stream)


def test_observe_only_empty_window_does_not_block_ready():
    gate = PromotionGate(opponents=("champion", "watch"),
                         observe_only=("watch",), window_games=1000)
    gate.record("champion", _normal(40, 100, 1000))
    assert gate.ready()  # watch has no data; readiness is over required opponents only
    assert gate.verdict()["promote"] is True


def test_observe_only_must_name_a_known_opponent():
    with pytest.raises(ValueError):
        PromotionGate(opponents=("champion",), observe_only=("typo",), window_games=10)


def test_observe_only_cannot_swallow_every_opponent():
    with pytest.raises(ValueError):
        PromotionGate(opponents=("champion",), observe_only=("champion",), window_games=10)


# --- Pooled Verdict (ADR-0040): margins accumulate across windows until a pooled
# CI decides; the pool survives holds and resets only on promotion. Exists because
# a per-window gate needs a true +~5 edge at n=8192 while a real KL-ball step is
# +0.5-2 — without pooling, provable-but-small gains can never bank.

def _edge(mean, sd, n):
    """Deterministic alternating margins with the given mean/sd (no RNG flake)."""
    return [mean + sd, mean - sd] * (n // 2)


def test_pooled_gate_banks_a_small_edge_that_no_single_window_could():
    gate = PromotionGate(window_games=2000, pooled=True)
    gate.record("champion", _edge(1.5, 50, 2000))   # SE ~1.1 -> CI_lo < 0
    v = gate.verdict()
    assert v["promote"] is False                     # one window can never bank +1.5
    gate.conclude(v)                                 # HOLD: pool must survive
    n_at_promote = None
    for _ in range(5):
        gate.record("champion", _edge(1.5, 50, 2000))
        v = gate.verdict()
        if v["promote"]:
            n_at_promote = v["opponents"]["champion"]["n"]
            break
        gate.conclude(v)
    # the pooled CI resolves after a few windows (SE shrinks ~1/sqrt(W))
    assert n_at_promote is not None and n_at_promote >= 4000


def test_pooled_gate_resets_only_on_promotion():
    gate = PromotionGate(window_games=100, pooled=True)
    gate.record("champion", _edge(50, 10, 100))      # decisive win
    v = gate.verdict()
    assert v["promote"] is True
    gate.conclude(v)
    assert gate.n("champion") == 0                   # promotion clears the pool


def test_unpooled_gate_conclude_resets_after_every_verdict():
    gate = PromotionGate(window_games=100)           # pooled defaults off
    gate.record("champion", _edge(0.0, 50, 100))
    v = gate.verdict()
    assert v["promote"] is False
    gate.conclude(v)
    assert gate.n("champion") == 0                   # old per-window behavior
