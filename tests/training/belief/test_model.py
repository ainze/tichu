"""BeliefModel + masked BCE loss + accuracy metric."""

import torch

from tichu_training.belief.model import BeliefModel, belief_accuracy, belief_loss


def test_forward_returns_3_by_56_logits_for_defaults():
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=64)
    x = torch.randn(7, 64)
    out = m(x)
    assert out.shape == (7, 3, 56)


def test_forward_respects_custom_dims():
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=32, num_opponents=2, num_cards=10, hidden=16)
    out = m(torch.randn(4, 32))
    assert out.shape == (4, 2, 10)


def test_loss_is_scalar_and_finite():
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=32, hidden=16)
    x = torch.randn(4, 32)
    labels = torch.zeros(4, 3, 56)
    mask = torch.ones(4, 3, 56, dtype=torch.bool)
    loss = belief_loss(m(x), labels, mask)
    assert loss.dim() == 0
    assert torch.isfinite(loss).all()


def test_loss_ignores_masked_out_positions():
    """Make the model output extreme garbage at masked-out positions; loss should still be finite and only reflect masked-in entries."""
    logits = torch.zeros(2, 3, 56)
    labels = torch.zeros(2, 3, 56)
    mask = torch.zeros(2, 3, 56, dtype=torch.bool)
    # Mark exactly one position in (batch 0, opp 0, card 0) as in-the-loss with label=1.
    mask[0, 0, 0] = True
    labels[0, 0, 0] = 1.0
    # Make logits huge negative at other positions (would dominate if not masked).
    logits[1, :, :] = -1e6
    loss = belief_loss(logits, labels, mask)
    assert torch.isfinite(loss).all()
    # The only contribution is one position: BCE with logit=0 and label=1.
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(0.0), torch.tensor(1.0)
    )
    assert abs(float(loss) - float(expected)) < 1e-6


def test_loss_near_zero_when_logits_match_labels():
    labels = torch.zeros(2, 3, 56)
    labels[0, 1, 5] = 1
    labels[1, 2, 17] = 1
    mask = torch.ones_like(labels, dtype=torch.bool)
    # Set logits to large positive where label=1, large negative elsewhere.
    logits = torch.where(labels == 1, torch.tensor(20.0), torch.tensor(-20.0))
    loss = belief_loss(logits, labels, mask)
    assert float(loss) < 1e-5


def test_accuracy_is_one_with_perfect_predictions():
    labels = torch.zeros(2, 3, 56)
    labels[0, 1, 5] = 1
    mask = torch.ones_like(labels, dtype=torch.bool)
    logits = torch.where(labels == 1, torch.tensor(5.0), torch.tensor(-5.0))
    acc = belief_accuracy(logits, labels, mask)
    assert acc == 1.0


def test_accuracy_only_counts_masked_in_positions():
    labels = torch.zeros(1, 3, 56)
    mask = torch.zeros(1, 3, 56, dtype=torch.bool)
    mask[0, 0, 0] = True  # only one position counts.
    # Logits agree at the masked-in position; disagree elsewhere.
    logits = torch.full_like(labels, fill_value=5.0)  # all predict True
    labels[0, 0, 0] = 1  # correct here
    acc = belief_accuracy(logits, labels, mask)
    assert acc == 1.0
