"""Preference Correction loss (ADR-0042).

The loss asserts a sign-verified ORDERING on one (chosen, alternative) pair per
row — never a target distribution over the Action Space. Tested on raw logit
tensors, no model and no parquet: the geometry is the specification.
"""

import pytest
import torch

from tichu_training.correction.loss import preference_loss


def _logits(rows, *, width=8):
    """(N, width) logits from a list of {index: value} dicts, default 0."""
    z = torch.zeros(len(rows), width)
    for i, row in enumerate(rows):
        for idx, val in row.items():
            z[i, idx] = val
    return z


def test_verified_non_correction_is_free_while_the_policy_is_right():
    """The barrier property — the entire justification for NOT rebalancing the
    ~17:1 class ratio (ADR-0042 §Decision 2). On a Verified Non-Correction the
    constraint z[chosen] >= z[alt] already holds at initialisation, because
    `chosen` IS the policy's legal-masked argmax. Such rows must contribute
    exactly zero, so they cost nothing until the positives drag them."""
    z = _logits([{2: 5.0, 6: 1.0}])  # chosen=2 well above alt=6
    loss = preference_loss(
        z,
        chosen_idx=torch.tensor([2]),
        alt_idx=torch.tensor([6]),
        is_correction=torch.tensor([False]),
        mean_delta=torch.tensor([0.0]),
        delta_scale=50.0,
    )
    assert loss.item() == 0.0


def _descend(z, *, steps=200, lr=0.1, **kw):
    """Plain gradient descent on the loss; returns the optimised logits."""
    z = z.clone().requires_grad_(True)
    opt = torch.optim.SGD([z], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        preference_loss(z, **kw).backward()
        opt.step()
    return z.detach()


def test_verified_correction_makes_the_alternative_the_argmax():
    """Descending on a Verified Correction must flip the ordering the tier-2
    verdict rejected: the alternative ends up ranked above the chosen action."""
    kw = dict(
        chosen_idx=torch.tensor([2]),
        alt_idx=torch.tensor([6]),
        is_correction=torch.tensor([True]),
        mean_delta=torch.tensor([50.0]),
        delta_scale=50.0,
    )
    z0 = _logits([{2: 5.0, 6: 1.0}])  # policy currently prefers chosen
    assert preference_loss(z0, **kw).item() > 0.0
    z1 = _descend(z0, **kw)
    assert int(z1[0].argmax()) == 6
    assert (z1[0, 6] - z1[0, 2]).item() >= 1.0  # cleared the verified margin


def test_margin_scales_with_the_verified_delta():
    """A correction worth +150 points should be asserted harder than one worth
    +20. The verified magnitude — not a flat label — sets the target gap, which
    is why the [50,150) and [150,400) Delta Bands are worth mining at all."""
    kw = dict(
        chosen_idx=torch.tensor([2, 2]),
        alt_idx=torch.tensor([6, 6]),
        is_correction=torch.tensor([True, True]),
        mean_delta=torch.tensor([20.0, 150.0]),
        delta_scale=50.0,
    )
    z = _descend(_logits([{2: 5.0, 6: 1.0}, {2: 5.0, 6: 1.0}]), **kw)
    small_gap = (z[0, 6] - z[0, 2]).item()
    large_gap = (z[1, 6] - z[1, 2]).item()
    assert large_gap > small_gap
    assert small_gap == pytest.approx(20.0 / 50.0, abs=0.05)
    assert large_gap == pytest.approx(150.0 / 50.0, abs=0.05)


def test_only_the_verified_pair_receives_gradient():
    """The structural difference from the killed CE mechanism. CE toward the
    alternative re-targets the whole legal distribution, so a few hundred rows
    generalise into a blanket rule over every similar state. This loss touches
    two logits and is silent about the other 1,807 — it cannot express a
    distribution shift, only an ordering the verifier signed off on."""
    z = _logits([{2: 5.0, 6: 1.0, 3: 4.9, 7: 4.8}]).requires_grad_(True)
    preference_loss(
        z,
        chosen_idx=torch.tensor([2]),
        alt_idx=torch.tensor([6]),
        is_correction=torch.tensor([True]),
        mean_delta=torch.tensor([50.0]),
        delta_scale=50.0,
    ).backward()
    touched = {i for i in range(z.shape[1]) if z.grad[0, i] != 0.0}
    assert touched == {2, 6}


def _mixed(n_neg):
    z = _logits([{2: 5.0, 6: 1.0}] * (n_neg + 1))
    kw = dict(
        chosen_idx=torch.tensor([2] * (n_neg + 1)),
        alt_idx=torch.tensor([6] * (n_neg + 1)),
        is_correction=torch.tensor([True] + [False] * n_neg),
        mean_delta=torch.tensor([50.0] + [0.0] * n_neg),
        delta_scale=50.0,
    )
    return z, kw


def test_carrying_more_non_corrections_does_not_weaken_the_corrections():
    """The barrier property, exactly rather than approximately. Normalising by
    row count would divide the corrections' gradient by the class ratio, so an
    `lr` swept on a [15,50)-only corpus would not transfer to the all-bands
    corpus ADR-0042 commits to. Satisfied non-corrections must be free of BOTH
    direction and scale."""
    # correction hinge = margin 1.0 - gap (1.0 - 5.0) = 5.0, over one correction
    for n_neg in (0, 1, 20, 500):
        z, kw = _mixed(n_neg)
        assert preference_loss(z, **kw).item() == pytest.approx(5.0)


def test_a_batch_of_only_non_corrections_is_well_defined():
    """~94% of screened trigger states are Verified Non-Corrections, so a
    correction-free minibatch is routine, not exotic. It must be finite, zero
    while the policy is right, and non-zero the moment the ordering the verifier
    upheld is violated — that is the barrier doing its job."""
    kw = dict(
        chosen_idx=torch.tensor([2, 2]),
        alt_idx=torch.tensor([6, 6]),
        is_correction=torch.tensor([False, False]),
        mean_delta=torch.tensor([0.0, 0.0]),
        delta_scale=50.0,
    )
    satisfied = preference_loss(_logits([{2: 5.0, 6: 1.0}] * 2), **kw)
    assert torch.isfinite(satisfied) and satisfied.item() == 0.0

    # the corrections have now dragged both rows past the barrier
    violated = preference_loss(_logits([{2: 1.0, 6: 2.5}] * 2), **kw)
    assert violated.item() == pytest.approx(3.0)  # 1.5 each, unnormalised
