"""v7 (ADR-0044): serving a mask-consuming policy.

`BCModel(use_legal_mask=True)` has a 3-argument forward and RAISES on a missing
mask — deliberately, so a caller that forgets it fails immediately instead of
serving a policy that sees 1809 zeros where training saw the legal set. That
contract has to survive the TorchScript round-trip and reach `MLAgent`, which is
what the co-train greedy gate drives on both sides of every window.

The mask fed at inference must match what training fed: the play-legality mask on
a Play Decision, ZEROS on wish / dragon rows (`bc.heads.play_mask_for_trunk`).
"""

import numpy as np
import torch

from tichu_engine.cards import Card, Suit
from tichu_engine.state import PrivateState, PublicState, Trick
from tichu_export.torchscript import exported_uses_legal_mask
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS
from tichu_training.bc.heads import BCModel
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION
from tichu_training.cli.export_model import export_policy_module


def _small(**kw) -> BCModel:
    torch.manual_seed(0)
    return BCModel(feature_dim=FEATURIZER_OUTPUT_DIM, trunk_hidden=16,
                   trunk_depth=1, trunk_out_dim=8, head_hidden=8, **kw)


def _private_state() -> PrivateState:
    hand = frozenset({Card(Suit.JADE, r) for r in range(2, 15)})
    public = PublicState(
        current_player=0, hand_sizes=(13, 13, 13, 13), scores=(0, 0),
        trick=Trick.empty(),
    )
    return PrivateState(player=0, hand=hand, public=public)


def test_export_records_whether_the_policy_consumes_the_mask(tmp_path):
    """The loader cannot introspect a traced module's arity, so the flag has to
    ride with the artifact — otherwise `MLAgent` has to guess, and guessing wrong
    means either a hard crash or (worse) silently feeding zeros."""
    with_mask = tmp_path / "with.pt"
    without = tmp_path / "without.pt"
    export_policy_module(_small(use_legal_mask=True), with_mask)
    export_policy_module(_small(use_legal_mask=False), without)

    assert exported_uses_legal_mask(with_mask) is True
    assert exported_uses_legal_mask(without) is False


def test_a_mask_consuming_export_survives_the_round_trip_and_acts(tmp_path):
    """End-to-end: trace, save, load, act. This is the path the greedy gate runs
    for the learner every window."""
    from tichu_inference.ml_agent import MLAgent

    path = tmp_path / "policy.pt"
    export_policy_module(_small(use_legal_mask=True), path)

    agent = MLAgent(str(path), skill_decile=9)
    action = agent.act(_private_state())
    assert action is not None
    assert agent.last_fallback_used is False, (
        "a mask-consuming policy must serve natively, not via the random-legal "
        "fallback — the fallback would silently mask a broken mask contract"
    )


def test_non_play_decisions_are_served_the_zero_mask(tmp_path):
    """Training fed zeros on wish / dragon rows (`play_mask_for_trunk`), so
    inference must too. Feeding a play-legality mask on a wish Decision would be
    a train/serve skew of exactly the kind ADR-0044 exists to avoid."""
    from tichu_inference.ml_agent import _trunk_legal_mask

    ps = _private_state()
    play_mask = _trunk_legal_mask(ps, is_play_decision=True)
    other_mask = _trunk_legal_mask(ps, is_play_decision=False)

    assert play_mask.shape == (1, HEAD_LOGIT_DIMS["play"])
    assert other_mask.shape == (1, HEAD_LOGIT_DIMS["play"])
    assert bool(play_mask.any()), "a Play Decision has legal Intents"
    assert not bool(other_mask.any()), "non-play rows saw zeros in training"
