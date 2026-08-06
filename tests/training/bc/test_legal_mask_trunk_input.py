"""v7 (ADR-0044 / PR #80): the legal-Intent mask as a TRUNK INPUT.

The mask has always been applied as an OUTPUT mask (`masked_cross_entropy`,
`sample_masked`); it never reached the trunk. PR #80 measured feeding it in at
**+3.04% relative NLL** on 300k decile-9 non-forced Play Decisions, with a
NEGATIVE (-2.24%) shuffled-mask placebo — so the gain is not a capacity effect.

It is a model-arch flag, NOT a featurizer section: the mask is already bit-packed
in the materialised bundle, so this costs no disk, no `FEATURIZER_VERSION` bump,
and no re-materialisation.

The concat lives INSIDE the model. Doing it at the call sites is the shape that
produced the shipped `team_scores` and schupfen `current_player` skews — four
call sites, four chances to disagree about the input contract.
"""

import pytest
import torch

from tichu_training.bc.heads import BCModel
from tichu_training.bc.decision_types import HEAD_LOGIT_DIMS

FEAT = 32
PLAY_LOGITS = HEAD_LOGIT_DIMS["play"]


def _model(**kw) -> BCModel:
    torch.manual_seed(0)
    return BCModel(feature_dim=FEAT, trunk_hidden=16, trunk_depth=1,
                   trunk_out_dim=8, head_hidden=8, **kw)


def _inputs(batch: int = 2):
    torch.manual_seed(1)
    return (
        torch.randn(batch, FEAT),
        torch.zeros(batch, dtype=torch.long),
    )


def test_the_mask_changes_the_output_when_the_flag_is_on():
    """The whole point: two identical states differing ONLY in which actions are
    legal must produce different logits. If they don't, the mask is being
    accepted and ignored — the failure mode a shape-only test would miss."""
    model = _model(use_legal_mask=True)
    features, skill = _inputs()

    mask_a = torch.zeros(2, PLAY_LOGITS, dtype=torch.bool)
    mask_a[:, :5] = True
    mask_b = torch.zeros(2, PLAY_LOGITS, dtype=torch.bool)
    mask_b[:, 100:130] = True

    out_a = model(features, skill, mask_a)["play"]
    out_b = model(features, skill, mask_b)["play"]
    assert not torch.allclose(out_a, out_b)


def test_a_missing_mask_fails_loudly_when_the_flag_is_on():
    """Silent zero-filling is the bug class this whole change is trying not to
    repeat. A caller that forgets the mask must break immediately, not serve a
    policy that quietly sees 1809 zeros where training saw the legal set."""
    model = _model(use_legal_mask=True)
    features, skill = _inputs()
    with pytest.raises(ValueError, match="legal_mask"):
        model(features, skill)


def test_the_flag_off_keeps_the_two_argument_contract():
    """v6-equivalent arm. Every existing call site passes two arguments, and the
    mask-off model must stay exactly that model — this is what makes the
    mask an independently switchable lever against the SAME bundle."""
    model = _model(use_legal_mask=False)
    features, skill = _inputs()
    baseline = model(features, skill)["play"]

    mask = torch.zeros(2, PLAY_LOGITS, dtype=torch.bool)
    mask[:, :5] = True
    assert torch.allclose(model(features, skill, mask)["play"], baseline)


def test_non_play_heads_contribute_zeros_not_their_own_narrow_mask():
    """Head masks are 1809 / 14 / 2 wide but the trunk input is fixed-width.

    A wish row's 14-wide mask must NOT be padded into play's action-index space —
    those indices mean entirely different actions, and the trunk would read a
    wish's legal ranks as a claim about Singles and Pairs. Zeros are the honest
    encoding: "no play-legality information here", which `phase` already implies.
    """
    from tichu_training.bc.heads import play_mask_for_trunk

    play_mask = torch.zeros(3, PLAY_LOGITS, dtype=torch.bool)
    play_mask[:, 7] = True
    assert torch.equal(play_mask_for_trunk("play", play_mask), play_mask)

    wish_mask = torch.zeros(3, HEAD_LOGIT_DIMS["wish"], dtype=torch.bool)
    wish_mask[:, 2] = True
    trunk_mask = play_mask_for_trunk("wish", wish_mask)
    assert trunk_mask.shape == (3, PLAY_LOGITS)
    assert not trunk_mask.any()
