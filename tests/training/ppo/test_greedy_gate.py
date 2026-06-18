"""Greedy-margin promotion gate source (option A, ADR-0034 follow-up).

The champion gate's old margin source was the SAMPLED PPO rollout reward, which
inflates as the policy sharpens (low-entropy learner barely hurt by sampling, the
higher-entropy BC opponent is) — a confound that drove 201 phantom promotions while
deployed (greedy) strength fell. The fix: source the gate's margins from a GREEDY
mini-tournament (the deployed `MLAgent` via `run_full_tournament`), exactly what
`check_cotrain` measures. These tests pin the new source's plumbing and its
faithfulness-by-construction (identical greedy policies tie every seat-swapped deal).
"""

from functools import partial

import numpy as np
import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.bc.call_model import GrandTichuCallNetwork, TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM as _D
from tichu_training.ppo.greedy_gate import (
    export_live_models,
    export_opponent,
    greedy_pair_margins,
    record_greedy_window,
)
from tichu_training.ppo.promotion_gate import PromotionGate
from tichu_training.ppo.rollout_parallel import save_rollout_weights
from tichu_training.awr.value_baseline import ValueBaseline
from tichu_training.perfect_info import PERFECT_INFO_DIM

_MODEL = dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16)
_ARCH = {
    "model": _MODEL,
    "schupfen_model": {"skill_dim": 8, "hidden": 16},
    "call_model": {"skill_dim": 8, "hidden": 16},
}


def _tiny_models(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "play": BCModel(_D, skill_buckets=10, **_MODEL),
        "schupfen": SchupfenNetwork(_D, skill_dim=8, hidden=16),
        "tichu": TichuCallNetwork(_D, skill_dim=8, hidden=16),
        "grand": GrandTichuCallNetwork(_D, skill_dim=8, hidden=16),
    }


def test_export_live_models_roundtrips_to_a_loadable_ml_agent(tmp_path):
    # export_live_models takes the LIVE learner nn.Modules (no .bin round-trip) and
    # produces TorchScript the MLAgent loads — the four files + the agent kwargs.
    from tichu_training.cli.check_cotrain import _build_agent

    kwargs = export_live_models(_tiny_models(), tmp_path / "learner")
    for name in ("policy.pt", "schupfen.pt", "tichu_call.pt", "grand_tichu_call.pt"):
        assert (tmp_path / "learner" / name).exists()
    agent = _build_agent("ml", skill_decile=9, **kwargs)
    # It plays a real Starting Position without falling back.
    pos = generate_full_position_pool(seed=0, n=1)[0]
    from tichu_eval.play_full import play_full_round
    play_full_round((agent, agent, agent, agent), pos.state, pos.grand_prefixes)


def test_identical_greedy_policies_net_to_zero_under_seat_swap(tmp_path):
    # Faithfulness by construction: two IDENTICAL greedy MLAgents play the SAME deal in
    # both seat arrangements, so each deal's two paired observations are exact negatives
    # (the team-0/team-1 deal asymmetry — the Mahjong-holder seat advantage, ~hundreds of
    # points — cancels) and the MEAN margin is exactly 0. This is the structural
    # anti-confound guarantee: a policy that is not actually stronger than the opponent
    # nets 0 greedily, no matter how large its SAMPLED rollout margin would have been.
    # (The large per-observation swing here vs the ~0 mean is also why a CI-usable greedy
    # gate needs many deals — the deal variance dwarfs the skill edge.)
    models = _tiny_models()
    kw = export_live_models(models, tmp_path / "same")
    positions = generate_full_position_pool(seed=1, n=4)
    margins = greedy_pair_margins(kw, kw, positions, skill_decile=9, workers=1)
    assert margins.shape == (2 * len(positions),)  # seat-swap => 2 obs per deal
    # Each seat-swapped pair (arrangement 1, arrangement 2) sums to zero.
    assert np.allclose(margins[0::2], -margins[1::2]), f"pairs must cancel, got {margins}"
    assert np.isclose(margins.mean(), 0.0), f"identical policies must net to 0, got {margins.mean()}"


def test_export_opponent_loads_rollout_weights_and_exports(tmp_path):
    # The pool opponents (champion / BC) are stored as rollout-weights .pt
    # ({"models": state_dicts, "critic": ...}); export_opponent rebuilds the arch,
    # loads those weights, and exports TorchScript identical in shape to the learner's.
    models = _tiny_models(seed=3)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16, depth=1)
    weights = tmp_path / "_champion.pt"
    save_rollout_weights(str(weights), models, critic)

    kwargs = export_opponent(str(weights), _ARCH, tmp_path / "opp")
    for name in ("policy.pt", "schupfen.pt", "tichu_call.pt", "grand_tichu_call.pt"):
        assert (tmp_path / "opp" / name).exists()

    # The exported opponent is byte-faithful to its source: greedy_pair_margins of the
    # live learner-equivalent vs this exported opponent matches exporting the SAME
    # models live (identical => all-zero margins).
    learner_kw = export_live_models(_tiny_models(seed=3), tmp_path / "learner3")
    positions = generate_full_position_pool(seed=2, n=3)
    margins = greedy_pair_margins(learner_kw, kwargs, positions, skill_decile=9, workers=1)
    # Byte-faithful opponent => identical policies => seat-swap pairs cancel, mean 0.
    assert np.allclose(margins[0::2], -margins[1::2])
    assert np.isclose(margins.mean(), 0.0)


def test_record_greedy_window_fills_every_opponents_window_in_one_eval(tmp_path):
    # record_greedy_window runs ONE mini-tournament of the live learner vs EACH pool
    # opponent and records 2*n_deals margins per opponent — so a single call fills the
    # gate's window for both, unlike the sampled gate's per-iter accumulation.
    learner = _tiny_models(seed=5)
    critic = ValueBaseline(PERFECT_INFO_DIM, hidden=16, depth=1)
    champ = tmp_path / "_champion.pt"
    bc = tmp_path / "_bc.pt"
    save_rollout_weights(str(champ), _tiny_models(seed=6), critic)
    save_rollout_weights(str(bc), _tiny_models(seed=7), critic)
    opp_cycle = [("champion", str(champ)), ("bc", str(bc))]

    n_deals = 3
    gate = PromotionGate(opponents=("champion", "bc"), window_games=2 * n_deals, seed=0)
    positions = generate_full_position_pool(seed=8, n=n_deals)
    record_greedy_window(
        gate, learner, opp_cycle, arch_cfg=_ARCH, positions=positions,
        skill_decile=9, workers=1, export_root=tmp_path / "_gate_export",
    )
    assert gate.n("champion") == 2 * n_deals
    assert gate.n("bc") == 2 * n_deals
    assert gate.ready()  # both windows full after a single eval
