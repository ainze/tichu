"""PPO Refine opponent league (ADR-0029): frozen-BC start + periodic frozen
snapshots of the improving learner, sampled as opponents."""

import random

import torch

from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.heads import HEAD_LOGIT_DIMS, BCModel
from tichu_training.ppo.league import League


def _tiny():
    model = BCModel(
        8, skill_dim=4, trunk_hidden=16, trunk_depth=1, trunk_out_dim=8, head_hidden=8
    )
    return model, ValueBaseline(8, hidden=8)


def test_league_samples_from_its_base_opponents():
    sentinel = object()
    league = League([sentinel])
    assert league.sample() is sentinel


def test_snapshot_grows_the_pool():
    model, critic = _tiny()
    league = League([object()])
    before = len(league)
    league.snapshot(model, critic, skill_decile=9)
    assert len(league) == before + 1


def test_league_is_bounded_to_keep_it_small():
    model, critic = _tiny()
    base = [object()]
    league = League(base, max_snapshots=2)
    for _ in range(5):
        league.snapshot(model, critic, skill_decile=9)
    # base opponents are always kept; snapshots are capped (oldest dropped).
    assert len(league) == len(base) + 2


def test_snapshot_is_frozen_and_independent_of_later_learner_updates():
    model, critic = _tiny()
    league = League([], max_snapshots=5)
    league.snapshot(model, critic, skill_decile=9)
    snap = league.sample()

    features = torch.randn(2, 8)
    skill = torch.full((2,), 9, dtype=torch.long)
    masks = torch.zeros(2, HEAD_LOGIT_DIMS["play"], dtype=torch.bool)
    masks[:, :4] = True

    before = snap.sample(features, skill, masks, generator=torch.Generator().manual_seed(0))
    # Drastically mutate the ORIGINAL learner after the snapshot was taken.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(100.0)
    after = snap.sample(features, skill, masks, generator=torch.Generator().manual_seed(0))

    assert torch.equal(before.indices, after.indices)
    assert torch.allclose(before.logprobs, after.logprobs)
