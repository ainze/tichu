"""Vine collection (ADR-0035): rows must be valid play-head transitions (legal
action, finite masked logprob, luck-free advantage), deterministic per seed, and
slot into `cotrain_update` as the play sub-batch. Tiny nets, no checkpoints."""

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.cli.train_cotrain import _build_models
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.perfect_info import PERFECT_INFO_DIM
from tichu_training.ppo.vine import collect_vine_rows, vine_net_batch

_ARCH = {
    "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
    "schupfen_model": {"skill_dim": 8, "hidden": 16},
    "call_model": {"skill_dim": 8, "hidden": 16},
}


def _models():
    torch.manual_seed(0)
    return _build_models(_ARCH)


def _rows(models, n_games=2, seed=5, **kwargs):
    positions = generate_full_position_pool(seed=31, n=n_games)
    return collect_vine_rows(models, positions, decisions_per_game=3, branches=3,
                             skill_decile=9, seed=seed, **kwargs)


def test_vine_rows_are_valid_play_transitions():
    models = _models()
    rows = _rows(models)
    assert 0 < len(rows) <= 6  # <= games * decisions_per_game
    for r in rows:
        assert r["features"].shape == (FEATURIZER_OUTPUT_DIM,)
        assert bool(r["mask"][r["action"]])  # the taken action is legal
        assert r["old_logp"] <= 0.0
        assert abs(r["advantage"]) < 1000.0


def test_vine_rows_are_deterministic_per_seed():
    a = _rows(_models(), seed=5)
    b = _rows(_models(), seed=5)
    assert len(a) == len(b)
    assert [r["action"] for r in a] == [r["action"] for r in b]
    assert [r["advantage"] for r in a] == [r["advantage"] for r in b]


def test_emit_branches_adds_positive_advantage_alternative_rows():
    # Branch emission (v1-autopsy enrichment): same playouts, the chosen row plus
    # one row per POSITIVE-advantage alternative. Negative-advantage alternatives
    # must NOT become rows — pushing down already-low-probability actions is the
    # unbounded burn-down that NaN'd vine v2's first run (play KL 0.01 -> 0.158,
    # then exp-overflow in the ratio).
    models = _models()
    chosen_only = _rows(models)
    rows = _rows(models, emit_branches=True)
    assert len(rows) > len(chosen_only)  # some better-than-baseline alts exist
    for r in rows:
        assert bool(r["mask"][r["action"]])
        assert r["old_logp"] <= 0.0
    # Rows sharing a feature vector are one decision's set: chosen first (any
    # advantage sign), every following alternative row strictly positive.
    by_decision: dict[bytes, list[float]] = {}
    for r in rows:
        by_decision.setdefault(r["features"].tobytes(), []).append(r["advantage"])
    for advs in by_decision.values():
        assert all(a > 0.0 for a in advs[1:])
    # The chosen-only rows are a subset: same decisions, same chosen advantages.
    assert [r["advantage"] for r in chosen_only] == [
        by_decision[r["features"].tobytes()][0] for r in chosen_only
    ]


def test_clipped_policy_loss_survives_extreme_old_logp():
    # Regression (vine v2 NaN crash): a dominated branch action collected with
    # old_logp ~ -90 must not overflow exp() into inf/NaN — loss and gradients
    # stay finite.
    from tichu_training.ppo.update import clipped_policy_loss

    new_logp = torch.tensor([-0.5, -2.0, -1.0], requires_grad=True)
    old_logp = torch.tensor([-90.0, -0.5, -1.0])
    advantages = torch.tensor([1.5, -2.0, 0.5])
    loss = clipped_policy_loss(new_logp, old_logp, advantages, clip_eps=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(new_logp.grad).all()


def test_min_abs_advantage_filters_near_ties():
    models = _models()
    rows = _rows(models, emit_branches=True)
    cutoff = sorted(abs(r["advantage"]) for r in rows)[len(rows) // 2] + 1e-6
    kept = _rows(models, emit_branches=True, min_abs_advantage=cutoff)
    assert 0 < len(kept) < len(rows)
    assert all(abs(r["advantage"]) >= cutoff for r in kept)
    expected = [r["advantage"] for r in rows if abs(r["advantage"]) >= cutoff]
    assert [r["advantage"] for r in kept] == expected


def test_vine_net_batch_shapes():
    rows = _rows(_models())
    nb = vine_net_batch(rows, skill_decile=9)
    n = len(rows)
    assert nb.features.shape == (n, FEATURIZER_OUTPUT_DIM)
    assert nb.actions.shape == (n,) and nb.actions.dtype == torch.long
    assert nb.masks.shape[0] == n and nb.masks.dtype == torch.bool
    assert nb.old_logp.shape == (n,) and nb.advantages.shape == (n,)


def test_vine_batch_drives_the_play_subupdate():
    # End-to-end: a main-rollout batch with its play group REPLACED by vine rows
    # must pass through cotrain_update and report play stats.
    import copy

    from tichu_eval.full_position_pool import generate_full_position_pool as gen
    from tichu_training.ppo.cotrain import (
        BatchedCoTrainPolicy,
        build_cotrain_batch,
        cotrain_update,
    )
    from tichu_training.ppo.rollout import collect_rollout

    models = _models()
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16)
    policy = BatchedCoTrainPolicy(
        models["play"], models["schupfen"], models["tichu"], models["grand"], critic,
        skill_decile=9, perfect_info=True,
        generator=torch.Generator().manual_seed(0), train_wish=True,
    )
    trajs = collect_rollout(gen(seed=7, n=2), policy, opponent_policy=policy,
                            learner_team=0)
    batch = build_cotrain_batch(trajs, skill_decile=9, gamma=1.0, lam=0.95)
    rows = _rows(models)
    batch.nets["play"] = vine_net_batch(rows, skill_decile=9)

    bc_models = {k: copy.deepcopy(m) for k, m in models.items()}
    optimizer = torch.optim.Adam(
        [{"params": m.parameters()} for m in models.values()]
        + [{"params": critic.parameters()}], lr=1e-4,
    )
    stats = cotrain_update(
        models, bc_models, critic, optimizer, batch,
        clip_eps=0.1, vf_coef=0.5,
        ent_coefs={k: 0.01 for k in ("play", "schupfen", "tichu", "grand", "wish")},
        kl_coefs={k: 1.0 for k in ("play", "schupfen", "tichu", "grand", "wish")},
        epochs=1,
    )
    assert "play_kl" in stats and "play_policy_loss" in stats
    assert all(torch.isfinite(torch.tensor(v)) for k, v in stats.items()
               if isinstance(v, float))


# --- Reference Field (ADR-0040, vine v3): a FROZEN second policy plays all four
# seats of every branch continuation — including a replay of the chosen action
# (the trunk-result parity trick is only valid when field == trunk policy).

def test_reference_field_identical_to_learner_reproduces_v2_exactly():
    # With reference weights == learner weights the chosen replay retraces the
    # deterministic trunk continuation (the miner parity invariant), so v3 must
    # be byte-equal to v2 — the reference split changes nothing but the field.
    models = _models()
    reference = _models()  # same seed => identical weights, separate modules
    base = _rows(models)
    same = _rows(models, reference_models=reference)
    assert [r["advantage"] for r in same] == [r["advantage"] for r in base]
    assert [r["action"] for r in same] == [r["action"] for r in base]


def test_reference_field_changes_returns_but_not_states_or_logprobs():
    # A DIFFERENT reference changes branch continuations (returns/advantages),
    # but the probed states come from the learner's trunk game and old_logp
    # stays the LEARNER's logprob of the action — containment semantics.
    models = _models()
    torch.manual_seed(123)
    reference = _build_models(_ARCH)  # different weights
    base = _rows(models)
    refd = _rows(models, reference_models=reference)
    assert [r["features"].tobytes() for r in refd] == \
           [r["features"].tobytes() for r in base]      # same trunk states
    assert [r["action"] for r in refd] == [r["action"] for r in base]
    assert [r["old_logp"] for r in refd] == [r["old_logp"] for r in base]
    assert [r["advantage"] for r in refd] != [r["advantage"] for r in base]


# --- Stratified decision selection (ADR-0040): the row budget prefers the states
# where the recoverable pool lives — hand <= 10 or a live caller — with uniform
# fallback (v1's near-tie dilution burned 3/4 of the budget on ~zero rows).

def _hand_size(features) -> int:
    return int(features[:56].round().sum())  # own_hand section


def _caller_live(features) -> bool:
    from tichu_training.featurizer import SECTION_OFFSETS
    t = SECTION_OFFSETS["tichu_callers"]
    g = SECTION_OFFSETS["grand_tichu_callers"]
    return float(features[t:t + 4].sum() + features[g:g + 4].sum()) > 0


def test_stratified_selection_prefers_pool_states():
    rows = _rows(_models(), stratify=True)
    assert rows
    assert all(_hand_size(r["features"]) <= 10 or _caller_live(r["features"])
               for r in rows), "with enough pool states, all picks must be pool states"


def test_stratified_budget_beyond_the_pool_falls_back_instead_of_truncating():
    # With a budget covering every eligible decision, stratify must merely
    # REORDER the picks (pool states first) — the probed set is identical to
    # uniform selection, proving nothing is dropped when the pool runs out.
    #
    # The budget must EXCEED the eligible count for that premise to hold, and the
    # eligible count is a property of the round the (randomly initialised) nets
    # happen to play — it moved 50 -> 53 when the featurizer widened at v7, which
    # silently turned this into a test of truncation order. 500 is far above any
    # single round's decision count, so the premise is now guaranteed rather than
    # coincidental.
    positions = generate_full_position_pool(seed=31, n=1)
    uni = collect_vine_rows(_models(), positions, decisions_per_game=500,
                            branches=3, skill_decile=9, seed=5)
    strat = collect_vine_rows(_models(), positions, decisions_per_game=500,
                              branches=3, skill_decile=9, seed=5, stratify=True)
    assert len(strat) == len(uni) > 0
    assert {r["features"].tobytes() for r in strat} == \
           {r["features"].tobytes() for r in uni}
