"""Tests for piKL — inference-time KL-regularized better-response (ADR-0037).

piKL re-weights the BC Anchor's action distribution by a paired-rollout
advantage, leashed toward the anchor by λ: ``π(a) ∝ τ(a)·exp(Q(a)/λ)``. These
tests pin the *contract* of the anchored softmax and the shared-world Q
aggregation — the pure pieces the ADR calls out as begging for unit tests
(λ→∞ = BC, support ⊆ τ, normalization, worlds shared across candidates).
"""

import random
from pathlib import Path

import numpy as np
import pytest

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_for_schupfen, deal_initial_state
from tichu_ml.rule_agent import RuleAgent
from tichu_training.search.blunder_miner import playout_from, team_relative
from tichu_training.search.determinize import sample_determinized_world
from tichu_training.search.pikl import (
    anchored_softmax,
    build_pikl_agent,
    intent_rank,
    piKLAgent,
    pikl_q,
    run_pikl_ab_shard,
    standardize_q,
)

_ITER06225 = Path("C:/workbench/tichu/data/runs/cotrain_wish_v5/export/iter_06225")


class _ScriptedAnchor(RuleAgent):
    """A RuleAgent (deterministic, full-round-capable — usable as the rollout
    field) that also exposes ``play_action_scores`` so it can stand in as a BC
    Anchor τ in tests, with a known descending τ over the legal actions."""

    def play_action_scores(self, private_state):
        legal = list(legal_actions_for(private_state))
        raw = np.linspace(1.0, 0.1, len(legal))
        probs = raw / raw.sum()
        return list(zip(legal, probs))


def _play_root(seed):
    """A real post-Schupfen Play root: the GameState, the actor's PrivateState,
    a frozen RuleAgent field, and 3 Determinized Worlds sampled from the root."""
    state = deal_initial_state(seed=seed)
    actor = state.public.current_player
    root_view = state.private_view(actor)
    field = [RuleAgent() for _ in range(4)]
    rng = random.Random(seed)
    worlds = [sample_determinized_world(root_view, None, rng) for _ in range(3)]
    return root_view, field, worlds


def test_anchored_softmax_returns_a_normalized_distribution():
    # Tracer bullet: π over the k candidates is a probability distribution.
    tau = np.array([0.5, 0.3, 0.2])
    q = np.array([10.0, -5.0, 2.0])
    pi = anchored_softmax(tau, q, lam=1.0)
    assert pi.shape == tau.shape
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi >= 0.0)


def test_large_lambda_recovers_the_bc_anchor():
    # λ→∞ is pure BC: the advantage is fully leashed away and π == τ.
    tau = np.array([0.5, 0.3, 0.2])
    q = np.array([10.0, -5.0, 2.0])
    pi = anchored_softmax(tau, q, lam=1e9)
    assert np.allclose(pi, tau)


def test_support_is_a_subset_of_the_anchor():
    # piKL can only re-rank what the anchor proposes: a candidate τ never proposes
    # (τ=0) gets zero mass however large its advantage — the method's hard ceiling.
    tau = np.array([0.6, 0.0, 0.4])
    q = np.array([0.0, 100.0, 0.0])  # the τ=0 candidate is by far the best
    pi = anchored_softmax(tau, q, lam=0.1)
    assert pi[1] == 0.0


def test_higher_advantage_gains_mass_relative_to_the_anchor():
    # The point of the search: a higher-Q candidate ends up with MORE mass than
    # the anchor gave it, a lower-Q one with less.
    tau = np.array([0.5, 0.5])
    q = np.array([3.0, -3.0])
    pi = anchored_softmax(tau, q, lam=1.0)
    assert pi[0] > tau[0]
    assert pi[1] < tau[1]


def test_huge_advantage_does_not_overflow_to_nan():
    # The ±20 clamp guards the float32 NaN vine hit at iter 169 (ADR-0035).
    tau = np.array([0.5, 0.5])
    q = np.array([1e6, 0.0])
    pi = anchored_softmax(tau, q, lam=1e-3)
    assert np.all(np.isfinite(pi))
    assert np.isclose(pi.sum(), 1.0)


def test_standardize_q_centers_and_scales_to_the_point_scale():
    # Raw Q is in Tichu points (±100/±200); standardizing to a fixed point-scale
    # makes Q/λ dimensionless so the paper's λ grid transfers (ADR-0037 D).
    q_raw = np.array([10.0, 20.0, 30.0])
    q_std = standardize_q(q_raw, scale=10.0)
    assert np.allclose(q_std, [-1.0, 0.0, 1.0])


def test_q_is_the_mean_team_relative_outcome_over_the_shared_worlds():
    # Q(a) = mean, over the passed-in worlds, of forcing a at the root then
    # playing the frozen field to terminal, team-relative to the actor.
    root_view, field, worlds = _play_root(seed=7)
    actor = root_view.player
    candidates = list(legal_actions_for(root_view))[:2]
    init = root_view.public.scores

    expected = [
        float(np.mean([
            team_relative(
                playout_from(field, w, forced_action=cand, initial_scores=init),
                actor,
            )
            for w in worlds
        ]))
        for cand in candidates
    ]
    assert pikl_q(field, root_view, candidates, worlds) == expected


def test_every_candidate_is_scored_over_the_same_worlds():
    # The shared-world guarantee, observed: a candidate's Q is independent of the
    # other candidates and of list order (no per-candidate resampling, no leakage).
    root_view, field, worlds = _play_root(seed=11)
    a, b = list(legal_actions_for(root_view))[:2]
    q = pikl_q(field, root_view, [a, a, b], worlds)
    assert q[0] == q[1]                       # duplicate candidate ⇒ identical Q
    q_rev = pikl_q(field, root_view, [b, a], worlds)
    assert q_rev[1] == q[0] and q_rev[0] == q[2]  # order-independent


def test_intent_rank_is_position_of_the_target_among_legal_by_logit():
    # The coverage pre-gate primitive: where does the anchor rank the better
    # Intent? 0 = top-1. Only the legal set competes; illegal logits are ignored.
    logits = np.zeros(1809)
    logits[5], logits[9], logits[2], logits[1] = 3.0, 2.0, 1.0, 0.0
    legal = [5, 2, 9, 1]  # desc by logit: 5, 9, 2, 1
    assert intent_rank(logits, legal, 5) == 0
    assert intent_rank(logits, legal, 9) == 1
    assert intent_rank(logits, legal, 1) == 3


def _pikl_agent(anchor, **kw):
    kw.setdefault("worlds", 3)
    kw.setdefault("k", 6)
    kw.setdefault("lam", 1.0)
    kw.setdefault("q_scale", 30.0)
    return piKLAgent(anchor, **kw)


def test_act_returns_a_legal_action_on_a_play_decision():
    # Tracer bullet: end-to-end act over candidates → Q → π → pick.
    state = deal_initial_state(seed=7)
    view = state.private_view(state.public.current_player)
    agent = _pikl_agent(_ScriptedAnchor())
    action = agent.act(view)
    assert action in set(legal_actions_for(view))


def test_large_lambda_plays_the_anchor_top_pick():
    # End-to-end λ→∞ = BC: the search is fully leashed, so piKL returns the
    # anchor's highest-τ candidate (slice-1's λ→∞=τ, all the way through act).
    state = deal_initial_state(seed=7)
    view = state.private_view(state.public.current_player)
    anchor = _ScriptedAnchor()
    agent = _pikl_agent(anchor, lam=1e9)
    assert agent.act(view) == anchor.play_action_scores(view)[0][0]


def test_non_play_decisions_delegate_to_the_anchor():
    # piKL touches Play only — a Schupfen pending decision rides the anchor.
    state = deal_for_schupfen(seed=7)
    view = state.private_view(state.public.current_player)
    anchor = _ScriptedAnchor()
    agent = _pikl_agent(anchor)
    assert agent.act(view) == anchor.act(view)


@pytest.mark.slow
def test_pikl_over_the_real_iter06225_anchor_plays_a_legal_action():
    # End-to-end on the actual τ: MLAgent anchor + frozen-field rollout → a legal
    # Play. The integration proof the stub tests can't give (real net, real Q).
    if not _ITER06225.exists():
        pytest.skip("iter_06225 export not present")
    agent = build_pikl_agent(export_dir=_ITER06225, worlds=3, k=6, lam=0.1)
    state = deal_initial_state(seed=7)
    view = state.private_view(state.public.current_player)
    assert agent.act(view) in set(legal_actions_for(view))


def test_ab_shard_skips_when_output_already_exists(tmp_path):
    # Resumability: a shard whose .npz exists returns False without touching the
    # export or running any tournament — a killed run re-runs only missing shards.
    out = tmp_path / "shard_0000.npz"
    out.write_bytes(b"done")
    assert run_pikl_ab_shard(
        export_dir="ignored-path", positions=[], out_path=out, hp={}
    ) is False
    assert out.read_bytes() == b"done"  # untouched
