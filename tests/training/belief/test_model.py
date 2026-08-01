"""BeliefModel + masked BCE loss + accuracy metric."""

import torch

from tichu_training.belief.model import (
    BeliefModel,
    belief_accuracy,
    belief_holder_loss,
    belief_loss,
    holder_probabilities,
)


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


def test_a_residual_belief_trunk_can_match_the_policy_trunk():
    """The width sweep found +0.024 flat across hidden 64-512, but this project's
    head-capacity probe found schupfen/tichu are *depth*-limited and width alone
    underperforms. So the Belief Model must be able to take the same trunk shape
    the play Policy Network uses (1024 wide, 4 residual blocks, 512 out)."""
    torch.manual_seed(0)
    m = BeliefModel(
        feature_dim=591, hidden=1024, depth=4, trunk_out_dim=512, residual=True,
    )
    assert m(torch.randn(3, 591)).shape == (3, 3, 56)

    from tichu_training.bc.model import TichuTrunk

    reference = TichuTrunk(in_dim=591, hidden=1024, depth=4, out_dim=512)
    trunk_params = sum(p.numel() for p in m.trunk.parameters())
    assert trunk_params == sum(p.numel() for p in reference.parameters())


def test_the_default_belief_model_is_unchanged():
    """`residual=False` must stay the 2-layer MLP so the existing sweep
    checkpoints keep loading — the additive-option discipline `SchupfenNetwork`
    already follows."""
    torch.manual_seed(0)
    m = BeliefModel(feature_dim=64, hidden=32)
    assert set(m.state_dict()) == {
        "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
        "head.weight", "head.bias",
    }


def test_per_card_softmax_makes_the_holder_distribution_coherent():
    """A card has exactly one holder, but independent sigmoids never learn that:
    the fitted model put 0.982 (sd 0.068) of mass on unseen cards and was off by
    up to 2.4 cards on the publicly-known hand sizes. Normalising over the three
    opponents makes the constraint hold by construction — and the world sampler
    consumes exactly these weights."""
    logits = torch.randn(5, 3, 56)
    probs = holder_probabilities(logits)

    assert probs.shape == (5, 3, 56)
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(5, 56))


def test_per_card_cross_entropy_scores_only_unseen_cards():
    """The loss counterpart: one 3-way term per Unseen Card, none for cards the
    actor can already see."""
    labels = torch.zeros(1, 3, 56)
    labels[0, 1, 5] = 1.0          # opponent 1 holds card 5
    labels[0, 2, 9] = 1.0          # opponent 2 holds card 9
    card_mask = labels.any(dim=1)  # (1, 56)

    confident = torch.where(labels == 1, torch.tensor(20.0), torch.tensor(-20.0))
    assert float(belief_holder_loss(confident, labels, card_mask)) < 1e-5

    # A predictor that names the wrong holder must be penalised...
    wrong = torch.zeros(1, 3, 56)
    wrong[0, 0, 5] = 20.0
    wrong[0, 0, 9] = 20.0
    assert float(belief_holder_loss(wrong, labels, card_mask)) > 1.0

    # ...but garbage on cards the actor can see must not count at all.
    noisy = confident.clone()
    seen = ~card_mask[0]
    noisy[0][:, seen] = 50.0
    torch.testing.assert_close(
        belief_holder_loss(noisy, labels, card_mask),
        belief_holder_loss(confident, labels, card_mask),
    )
