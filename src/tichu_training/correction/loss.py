"""The Preference Correction loss ([ADR-0042](docs/adr/0042-preference-correction.md)).

One row per tier-2 verdict. The loss constrains exactly ONE pair of Intent logits
— `(chosen, alternative)` — and says nothing about the other 1,807. That is the
whole design: the calibration 15 inference-time interventions certified is what
this loss is *silent about*, not what it overwrites. Hard CE toward the verified
alternative (the mechanism killed 2026-06-11) instead re-targets the full legal
distribution, which is how a few hundred rows became a blanket rule.
"""

import torch


def preference_loss(
    logits: torch.Tensor,
    *,
    chosen_idx: torch.Tensor,
    alt_idx: torch.Tensor,
    is_correction: torch.Tensor,
    mean_delta: torch.Tensor,
    delta_scale: float,
) -> torch.Tensor:
    """Mean hinge loss over the verified `(chosen, alternative)` orderings.

    `logits` is `(N, action_space)`; every other argument is `(N,)`.
    `is_correction` selects the verdict: True = **Verified Correction** (the
    alternative beat the chosen action across Determinized Worlds), False =
    **Verified Non-Correction** (screened, did not beat it).

    A Verified Correction wants `z[alt] > z[chosen]` by a margin of
    `mean_delta / delta_scale` — the verified magnitude sets how hard the row
    pushes. A Verified Non-Correction wants only `z[chosen] >= z[alt]`, at **zero
    margin**: `chosen` is already the legal-masked argmax, so those rows start
    satisfied and contribute no gradient until the corrections drag them. A
    positive margin there would make them push on their own and destroy that
    barrier property (ADR-0042 §Decision 2).
    """
    rows = torch.arange(logits.shape[0], device=logits.device)
    z_chosen = logits[rows, chosen_idx]
    z_alt = logits[rows, alt_idx]

    margin = torch.where(
        is_correction, mean_delta / delta_scale, torch.zeros_like(mean_delta)
    )
    gap = torch.where(is_correction, z_alt - z_chosen, z_chosen - z_alt)
    hinge = torch.relu(margin - gap)

    # Normalised by the CORRECTION count, not the row count. Dividing by rows
    # would shrink every step in proportion to how many Verified Non-Corrections
    # the batch happens to carry, so `delta_scale` and `lr` swept on one corpus
    # would not transfer to a corpus with a different class ratio. Here a
    # satisfied non-correction is free of both direction and scale, and only
    # starts costing once the corrections have dragged it into violation.
    n_corrections = is_correction.sum().clamp(min=1)
    return hinge.sum() / n_corrections
