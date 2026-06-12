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


def test_emit_branches_adds_valid_alternative_rows():
    # All-branch emission (v1-autopsy enrichment): same playouts, one row per
    # branch. Branch rows must be legal play transitions, and within a decision
    # the advantages sum to zero (each is rel - mean over the branch set).
    models = _models()
    chosen_only = _rows(models)
    rows = _rows(models, emit_branches=True)
    assert len(rows) > len(chosen_only)  # alternatives became rows
    for r in rows:
        assert bool(r["mask"][r["action"]])
        assert r["old_logp"] <= 0.0
    # Consecutive rows sharing a feature vector are one decision's branch set.
    by_decision: dict[bytes, list[float]] = {}
    for r in rows:
        by_decision.setdefault(r["features"].tobytes(), []).append(r["advantage"])
    for advs in by_decision.values():
        assert abs(sum(advs)) < 1e-3
    # The chosen-only rows are a subset: same decisions, same chosen advantages.
    assert [r["advantage"] for r in chosen_only] == [
        advs[0] for advs in
        ([by_decision[r["features"].tobytes()] for r in chosen_only])
    ]


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
