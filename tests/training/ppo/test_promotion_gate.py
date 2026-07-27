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


def test_pooled_hold_does_not_redraw_until_new_margins_arrive():
    # Regression (live vine v3, iter 128-134): the trainer polls ready() every
    # iteration; historically reset() emptied the pool after each verdict, but a
    # pooled HOLD keeps the margins — without a new-data latch the same verdict
    # re-draws (and re-logs) every iteration until the next greedy window.
    gate = PromotionGate(window_games=100, pooled=True)
    gate.record("champion", _edge(0.0, 50, 100))
    v = gate.verdict()
    assert v["promote"] is False
    gate.conclude(v)
    assert not gate.ready(), "concluded hold must not re-arm without new margins"
    gate.record("champion", _edge(0.0, 50, 100))  # the next window's data arrives
    assert gate.ready()
    assert gate.verdict()["opponents"]["champion"]["n"] == 200  # still pooled


# --- seat-swap cluster bootstrap (`paired`) ---------------------------------
# The greedy window records 2*n_deals values: the two seat arrangements of each
# deal, ADJACENT. Card luck enters them with opposite sign, so it cancels within
# the pair; resampling them flat re-counts uncertainty the design already removed.


def _swapped(edge, luck_sd, n_deals, seed=0):
    """n_deals worth of seat-swap deltas, flattened pair-adjacent like
    `_play_pair_full`. Each deal contributes (edge + luck, edge - luck)."""
    rng = np.random.default_rng(seed)
    luck = rng.normal(0.0, luck_sd, n_deals)
    skill = rng.normal(edge, 5.0, n_deals)
    out = []
    for s, l in zip(skill, luck):
        out.extend((s + l, s - l))
    return out


def test_paired_ci_is_narrower_but_same_point_estimate():
    margins = _swapped(edge=3.0, luck_sd=200.0, n_deals=2000)
    flat = PromotionGate(window_games=4000, paired=False)
    paired = PromotionGate(window_games=4000, paired=True)
    flat.record("champion", margins)
    paired.record("champion", margins)
    f, p = _champ(flat.verdict()), _champ(paired.verdict())

    assert f["n"] == p["n"] == 4000                  # reported n is the raw obs count
    assert f["mean"] == pytest.approx(p["mean"])     # point estimate is untouched
    f_hw = (f["ci_hi"] - f["ci_lo"]) / 2
    p_hw = (p["ci_hi"] - p["ci_lo"]) / 2
    assert p_hw < f_hw / 2, "cancelling luck must collapse when pairs are clustered"


def test_paired_gate_promotes_an_edge_the_flat_gate_cannot_see():
    # The live symptom: a real +3/deal edge buried under seat-swap-cancelling luck.
    margins = _swapped(edge=3.0, luck_sd=200.0, n_deals=2000, seed=7)
    flat = PromotionGate(window_games=4000, paired=False)
    paired = PromotionGate(window_games=4000, paired=True)
    flat.record("champion", margins)
    paired.record("champion", margins)
    assert flat.verdict()["promote"] is False
    assert paired.verdict()["promote"] is True


def test_paired_still_rejects_a_genuinely_zero_edge():
    # Narrower bars must not become a rubber stamp.
    margins = _swapped(edge=0.0, luck_sd=200.0, n_deals=2000, seed=3)
    gate = PromotionGate(window_games=4000, paired=True)
    gate.record("champion", margins)
    v = gate.verdict()
    assert _champ(v)["ci_lo"] < 0.0 < _champ(v)["ci_hi"]
    assert v["promote"] is False


def test_paired_pools_across_windows_keeping_pairs_aligned():
    # Pooled mode concatenates windows; each is an even, pair-adjacent block, so
    # the clustering must stay aligned across the join. If it drifted by one, pairs
    # would straddle deals and the cancelling luck would stop cancelling — so the
    # pooled paired CI staying tight IS the alignment assertion.
    paired = PromotionGate(window_games=3000, paired=True, pooled=True)
    flat = PromotionGate(window_games=3000, paired=False, pooled=True)
    for w in range(3):
        window = _swapped(edge=2.0, luck_sd=200.0, n_deals=500, seed=w)
        paired.record("champion", window)
        flat.record("champion", window)
    assert paired.n("champion") == 3000
    p, f = _champ(paired.verdict()), _champ(flat.verdict())
    assert p["mean"] == pytest.approx(f["mean"])
    assert (p["ci_hi"] - p["ci_lo"]) < (f["ci_hi"] - f["ci_lo"]) / 2


def test_paired_rejects_an_odd_number_of_margins():
    # An odd count means a Position's second arrangement is missing — silently
    # pairing across deals would be worse than failing.
    gate = PromotionGate(window_games=4, paired=True)
    gate.record("champion", [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="even number of seat-swap deltas"):
        gate.verdict()


def test_pooled_paired_accumulates_independently_per_opponent():
    # Each opponent is a DIFFERENT comparison (learner-vs-champion, learner-vs-bc),
    # so their pools must never mix: pooling them into one stream would average two
    # unrelated questions. Drive one opponent at a real edge and one at zero — if the
    # pools leaked, both means would drift toward the average of the two.
    gate = PromotionGate(opponents=("champion", "bc"), window_games=1000,
                         paired=True, pooled=True)
    for w in range(3):
        gate.record("champion", _swapped(edge=0.0, luck_sd=200.0, n_deals=500, seed=w))
        gate.record("bc", _swapped(edge=40.0, luck_sd=200.0, n_deals=500, seed=100 + w))
        v = gate.verdict()
        assert v["promote"] is False, "champion sits at 0 — beat-ALL must hold"
        gate.conclude(v)

    assert gate.n("champion") == 3000
    assert gate.n("bc") == 3000
    opp = gate.verdict()["opponents"]
    assert opp["champion"]["mean"] == pytest.approx(0.0, abs=2.0)
    assert opp["bc"]["mean"] == pytest.approx(40.0, abs=2.0)
    assert opp["champion"]["ci_lo"] < 0.0 < opp["champion"]["ci_hi"]
    assert opp["bc"]["ci_lo"] > 0.0


def test_promotion_clears_every_opponents_pool_not_just_the_deciding_one():
    # The champion changes on promotion, so its pool is stale — but so is every other
    # stream's, because they all measured a learner that has now been elevated. A
    # partial clear would carry pre-promotion margins into the next window.
    gate = PromotionGate(opponents=("champion", "bc"), window_games=1000,
                         paired=True, pooled=True)
    gate.record("champion", _swapped(edge=30.0, luck_sd=200.0, n_deals=500, seed=1))
    gate.record("bc", _swapped(edge=30.0, luck_sd=200.0, n_deals=500, seed=2))
    v = gate.verdict()
    assert v["promote"] is True
    gate.conclude(v)
    assert gate.n("champion") == 0
    assert gate.n("bc") == 0


def test_pooled_is_not_ready_until_every_required_opponent_has_a_window():
    # Opponents can accumulate at different rates (the SAMPLED path alternates the
    # rollout opponent per iteration). A full pool on one stream must not arm the
    # verdict while another is still short.
    gate = PromotionGate(opponents=("champion", "bc"), window_games=1000,
                         paired=True, pooled=True)
    gate.record("champion", _swapped(edge=30.0, luck_sd=50.0, n_deals=1000, seed=1))
    assert gate.n("champion") == 2000
    assert not gate.ready(), "bc has no margins yet — cannot draw a beat-ALL verdict"
    assert gate.verdict()["promote"] is False
    gate.record("bc", _swapped(edge=30.0, luck_sd=50.0, n_deals=500, seed=2))
    assert gate.ready()


def test_observe_only_pool_never_blocks_a_pooled_promotion():
    # cpfix3328 is observe-only and currently NEGATIVE (-1.50 live). Its pool must
    # accumulate and report, but never gate readiness or the promote decision.
    gate = PromotionGate(opponents=("champion", "cpfix3328"), window_games=1000,
                         observe_only=("cpfix3328",), paired=True, pooled=True)
    gate.record("champion", _swapped(edge=30.0, luck_sd=200.0, n_deals=500, seed=1))
    gate.record("cpfix3328", _swapped(edge=-20.0, luck_sd=200.0, n_deals=500, seed=2))
    v = gate.verdict()
    assert v["opponents"]["cpfix3328"]["mean"] == pytest.approx(-20.0, abs=3.0)
    assert v["opponents"]["cpfix3328"]["ci_hi"] < 0.0, "observe stream is clearly negative"
    assert v["promote"] is True, "observe-only must not veto"
