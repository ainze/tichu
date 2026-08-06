"""v7 (ADR-0044): widening a v6 net into v7 shape is EXACTLY function-preserving.

ADR-0044 retrains only `BCModel` and the Schupfen Network at v7; the two Call
Networks are widened instead. That is licensed by a proof, not a guess — Grand
Tichu is called on the first eight cards and Schupfen is the pre-play pass, so
the 233-dim Rich History Block is identically zero at both (pinned in
`tests/training/featurizer/test_rich_history_section.py`). Widening therefore
cannot change what those nets compute.

These tests state that as BEHAVIOUR: the widened net, fed a v7 vector, returns
what the original returned on its v6 prefix. Comparing weight tensors instead
would pass while the column PERMUTATION was wrong — and it is easy to get wrong,
because the pre-projection layout is `[features, skill_emb]`, so v6's skill
columns sit at 591:655 and must move to 824:888.
"""

import torch

from tichu_training.bc.call_model import TichuCallNetwork
from tichu_training.bc.heads import BCModel
from tichu_training.bc.schupfen_model import SchupfenNetwork
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
from tichu_training.widen import widen_to_current_featurizer

V6_DIM = 591


def _v7_vector_from(v6: torch.Tensor) -> torch.Tensor:
    """A v7 Feature Vector whose v6 prefix is `v6` and whose Rich History Block
    is zero — exactly what Schupfen and Grand Tichu see at v7."""
    pad = torch.zeros(v6.shape[0], FEATURIZER_OUTPUT_DIM - V6_DIM)
    return torch.cat([v6, pad], dim=-1)


def test_widened_call_network_is_function_identical():
    torch.manual_seed(0)
    old = TichuCallNetwork(V6_DIM, skill_dim=8, hidden=16)
    old.eval()
    features = torch.randn(4, V6_DIM)
    skill = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        expected = old(features, skill)

    new = TichuCallNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    widen_to_current_featurizer(old, new, old_feature_dim=V6_DIM)
    new.eval()
    with torch.no_grad():
        got = new(_v7_vector_from(features), skill)

    torch.testing.assert_close(got, expected)


def test_widened_schupfen_network_is_function_identical():
    torch.manual_seed(0)
    old = SchupfenNetwork(V6_DIM, skill_dim=8, hidden=16)
    old.eval()
    features = torch.randn(3, V6_DIM)
    skill = torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        expected = old(features, skill)

    new = SchupfenNetwork(FEATURIZER_OUTPUT_DIM, skill_dim=8, hidden=16)
    widen_to_current_featurizer(old, new, old_feature_dim=V6_DIM)
    new.eval()
    with torch.no_grad():
        got = new(_v7_vector_from(features), skill)

    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e)


def test_widened_bc_model_is_function_identical():
    torch.manual_seed(0)
    kw = dict(trunk_hidden=16, trunk_depth=1, trunk_out_dim=8, head_hidden=8,
              skill_dim=8)
    old = BCModel(feature_dim=V6_DIM, **kw)
    old.eval()
    features = torch.randn(2, V6_DIM)
    skill = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        expected = old(features, skill)["play"]

    new = BCModel(feature_dim=FEATURIZER_OUTPUT_DIM, **kw)
    widen_to_current_featurizer(old, new, old_feature_dim=V6_DIM)
    new.eval()
    with torch.no_grad():
        got = new(_v7_vector_from(features), skill)["play"]

    torch.testing.assert_close(got, expected)


def test_widening_into_a_mask_consuming_model_zeroes_the_mask_columns():
    """Warm-starting the v7 lineage from a mask-less v6 Checkpoint must ignore
    the mask at init — otherwise the "identical at init" claim behind the
    pre-registered cpfix3328 fallback is false, and the run does not start at
    parity with the champion after all."""
    torch.manual_seed(0)
    kw = dict(trunk_hidden=16, trunk_depth=1, trunk_out_dim=8, head_hidden=8,
              skill_dim=8)
    old = BCModel(feature_dim=V6_DIM, **kw)
    old.eval()
    features = torch.randn(2, V6_DIM)
    skill = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        expected = old(features, skill)["play"]

    new = BCModel(feature_dim=FEATURIZER_OUTPUT_DIM, use_legal_mask=True, **kw)
    widen_to_current_featurizer(old, new, old_feature_dim=V6_DIM)
    new.eval()

    from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS
    for pattern in (0.0, 1.0):  # any mask whatsoever must be ignored
        mask = torch.full((2, HEAD_LOGIT_DIMS["play"]), pattern)
        with torch.no_grad():
            got = new(_v7_vector_from(features), skill, mask)["play"]
        torch.testing.assert_close(got, expected)
