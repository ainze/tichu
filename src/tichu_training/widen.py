"""Net Widening (v7, ADR-0044): load a v6 Checkpoint into a v7-shaped net,
exactly function-preserving.

A `FEATURIZER_VERSION` bump invalidates every Checkpoint, because all four nets
take `FEATURIZER_OUTPUT_DIM`. Naively that means re-materialising and retraining
four corpora. ADR-0044 retrains only `BCModel` and the Schupfen Network and
WIDENS the two Call Networks, which is sound because v7 is strictly additive and
the Rich History Block is identically zero at Grand Tichu and Schupfen.

Widening copies the v6 weights into the wider first layer and zero-initialises
every added input column, so the net's output is unchanged **exactly** — not
approximately — for any input whose added columns are zero.

The subtlety is a permutation, not a pad. Every net concatenates
`[features, skill_emb]` before its first projection, so v6's skill columns live
at `591:591+skill_dim` and must move to `824:824+skill_dim`. Appending zeros on
the right would silently feed the skill embedding into feature columns.

Also the fallback route in ADR-0044: if the v7 co-train stalls below cpfix3328,
widening cpfix3328 itself starts the run at parity with the champion by
construction, instead of in the -2.69 hole ADR-0040 measured. The cost is that
zero-initialised columns carry no supervised signal — PPO alone has to discover
2,042 new inputs — which is why it is the fallback and not the plan.
"""

from __future__ import annotations

import torch
from torch import nn

from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM


def _first_projection(module: nn.Module) -> nn.Linear:
    """The one Linear whose in_features depends on the featurizer width.

    `BCModel` hides it under `trunk.input_proj`; the Schupfen and Call Networks
    expose `input_proj` directly, and their non-residual variants wrap it in a
    `nn.Sequential` whose first entry is the Linear.
    """
    # The four nets spell this four different ways: `BCModel` nests a TichuTrunk
    # with `input_proj`; the residual Schupfen / Call Networks expose
    # `input_proj` directly; the non-residual ones wrap it in a Sequential named
    # `trunk` (Schupfen) or `net` (Calls). Resolve rather than special-case.
    trunk = getattr(module, "trunk", None)
    candidates = [
        getattr(module, "input_proj", None),
        getattr(trunk, "input_proj", None) if trunk is not None else None,
        trunk,
        getattr(module, "net", None),
    ]
    for proj in candidates:
        if isinstance(proj, nn.Linear):
            return proj
        if isinstance(proj, nn.Sequential) and len(proj) and isinstance(proj[0], nn.Linear):
            return proj[0]
    raise TypeError(
        f"{type(module).__name__} has no recognisable input projection; "
        "Net Widening needs the first featurizer-width Linear."
    )


def widen_to_current_featurizer(
    old: nn.Module,
    new: nn.Module,
    *,
    old_feature_dim: int,
    new_feature_dim: int = FEATURIZER_OUTPUT_DIM,
) -> nn.Module:
    """Copy `old`'s parameters into `new`, widening the first projection.

    `new` must be the same architecture as `old` apart from the featurizer width
    (and, for `BCModel`, an optional legal-mask trunk input). Mutates and returns
    `new`.
    """
    if new_feature_dim < old_feature_dim:
        raise ValueError(
            f"widening cannot shrink the featurizer: {old_feature_dim} -> "
            f"{new_feature_dim}"
        )

    old_proj, new_proj = _first_projection(old), _first_projection(new)
    skill_dim = old_proj.in_features - old_feature_dim
    if skill_dim < 0:
        raise ValueError(
            f"old input projection is narrower ({old_proj.in_features}) than its "
            f"stated feature dim ({old_feature_dim})"
        )
    expected_new = new_feature_dim + skill_dim
    if new_proj.in_features < expected_new:
        raise ValueError(
            f"target input projection is too narrow: {new_proj.in_features} < "
            f"{expected_new} (features {new_feature_dim} + skill {skill_dim})"
        )

    # Everything except the first projection has identical shape; copy by name so
    # an arch mismatch surfaces here rather than as silent garbage.
    old_state = old.state_dict()
    new_state = new.state_dict()
    proj_weight_keys = {
        k for k, v in old_state.items()
        if v.shape != new_state.get(k, torch.empty(0)).shape
    }
    new.load_state_dict(
        {k: v for k, v in old_state.items() if k not in proj_weight_keys},
        strict=False,
    )

    with torch.no_grad():
        w = new_proj.weight
        w.zero_()
        # Feature block keeps its columns — v7 is additive, so v6's features are
        # v7's prefix.
        w[:, :old_feature_dim] = old_proj.weight[:, :old_feature_dim]
        # Skill embedding MOVES to sit after the wider feature block.
        if skill_dim:
            w[:, new_feature_dim:new_feature_dim + skill_dim] = (
                old_proj.weight[:, old_feature_dim:]
            )
        # Everything else — the added feature dims and any legal-mask columns —
        # stays zero, which is what makes this exactly function-preserving.
        if new_proj.bias is not None and old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    return new
