"""File-based opponent league for co-training (ADR-0034 #1, league lever).

Unlike the play-only in-memory League, co-training opponents run in spawned rollout
workers, so the league holds weight-FILE paths: a fixed frozen-BC base plus periodic
frozen learner snapshots (oldest evicted). It round-trips through the Resume Bundle.
"""

import random

import torch

from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.ppo.cotrain_league import CoTrainLeague
from tichu_training.ppo.rollout_parallel import save_rollout_weights

_D = 8


def _models_and_critic():
    models = {
        "play": BCModel(_D, skill_dim=8, trunk_hidden=16, trunk_depth=1, trunk_out_dim=8, head_hidden=8),
        "schupfen": SchupfenNetwork(_D, skill_dim=8, hidden=8),
        "tichu": TichuCallNetwork(_D, skill_dim=8, hidden=8),
        "grand": GrandTichuCallNetwork(_D, skill_dim=8, hidden=8),
    }
    return models, ValueBaseline(_D, hidden=8)


def _base(tmp_path):
    models, critic = _models_and_critic()
    base = tmp_path / "base_bc.pt"
    save_rollout_weights(str(base), models, critic)
    return str(base)


def test_league_seeds_with_base_and_samples_it(tmp_path):
    league = CoTrainLeague(tmp_path / "league", _base(tmp_path), max_snapshots=3,
                           rng=random.Random(0))
    # With only the base present, every sample is the frozen-BC bundle.
    assert league.members() == [league._base[0]]
    assert league.sample() == league._base[0]


def test_league_snapshots_grow_then_evict_oldest_keeping_base(tmp_path):
    base = _base(tmp_path)
    league = CoTrainLeague(tmp_path / "league", base, max_snapshots=2, rng=random.Random(0))
    models, critic = _models_and_critic()

    paths = []
    for it in (10, 20, 30):  # 3 snapshots, cap 2 -> oldest (iter 10) evicted
        league.snapshot(models, critic, it)
        paths.append(str((tmp_path / "league" / f"snap_iter_{it:05d}.pt")))

    members = league.members()
    assert base in members                       # base never evicted
    assert paths[0] not in members               # oldest snapshot dropped
    assert paths[1] in members and paths[2] in members
    assert not (tmp_path / "league" / "snap_iter_00010.pt").exists()  # file deleted


def test_league_state_round_trips_for_resume(tmp_path):
    base = _base(tmp_path)
    league = CoTrainLeague(tmp_path / "league", base, max_snapshots=3, rng=random.Random(0))
    models, critic = _models_and_critic()
    league.snapshot(models, critic, 5)

    restored = CoTrainLeague.from_state(league.state(), league_dir=tmp_path / "league",
                                        rng=random.Random(0))
    assert restored.members() == league.members()
