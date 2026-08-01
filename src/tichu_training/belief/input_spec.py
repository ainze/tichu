"""Belief input spec — the identity of the vector the Belief Model consumes.

At v6 the belief input **is** the policy Feature Vector, unextended:

    belief_input = featurize(private_view(seat))        # FEATURIZER_OUTPUT_DIM

[ADR-0028](../../docs/adr/0028-belief-input-compressed-history-projections.md)
gave Belief its own input = policy features + a compact History block of
per-opponent projections (H1 declined_top, H2 lead_summary, H3 pass_pressure,
H4 play_time), and its ablation settled on **B-core** (policy + H1 + H2 + H3);
H4 play-time was *rejected* (B-full < B-core).

[ADR-0038](../../docs/adr/0038-featurizer-v6-played-by-and-schupfen-received.md)
then folded H1/H2/H3 into the policy featurizer itself as the `declined_top` /
`lead_summary` / `pass_pressure` sections, fed by live per-seat engine
accumulators. So at v6 the policy Feature Vector already *is* B-core, and the
belief-side History block is pure duplication — dropped, along with the
three-tier prefix machinery whose ablation ADR-0028 already decided.

Belief keeps its own version stamp (`BELIEF_INPUT_VERSION`) and its own
`feature_dim` / continuous-column set in the bundle manifest so a belief-input
change can invalidate belief bundles independently of the policy featurizer.
"""

from __future__ import annotations

from tichu_training.featurizer import (
    CONTINUOUS_FEATURE_COLUMNS, FEATURIZER_OUTPUT_DIM,
)


# v2: the belief input is the bare policy Feature Vector again — v6 absorbed
# the ADR-0028 B-core History channels, and H4 play-time was ablated out. v1
# was policy + the 83-dim History block.
BELIEF_INPUT_VERSION = "v2"

# The Belief Model's input width. Tracks the policy featurizer by construction;
# `tests/training/belief/test_input_spec.py` pins the two together so a
# featurizer bump fails loudly instead of silently mis-shaping belief inputs.
BELIEF_FEATURE_DIM = FEATURIZER_OUTPUT_DIM


def belief_continuous_columns() -> tuple[int, ...]:
    """Continuous (non-bit-packable) column indices of the belief input. With
    no History suffix these are exactly the policy's continuous columns."""
    return tuple(CONTINUOUS_FEATURE_COLUMNS)
