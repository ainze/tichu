"""v7 (ADR-0044) is strictly ADDITIVE: `featurize(...)[:591]` is the v6 Feature
Vector, bit for bit.

This is not a tidiness preference — two things depend on it:

  * **v6-View Prefix.** `ppo/cotrain.py` featurizes ONCE per Decision and shares
    the tensor across seats, and `ppo/greedy_gate.py` rebuilds pool opponents
    through a single live `arch_cfg`. Prefix-compatibility is what lets the served
    `cpfix3328` Checkpoint keep playing in the v7 rollout and promotion gate
    behind a `feats[:, :591]` slice, instead of a frozen second featurizer plus a
    second `featurize()` call in the rollout's hottest loop.
  * **The control arm.** A v6-equivalent BC arm is `[:591]` against the same
    bundle — provable, not asserted, and runnable without re-materialising.

The golden was captured from the v6 code itself (commit prior to the v7 bump) over
states that populate every v6 section; see `v6_golden_states`. Regenerating it is a
deliberate act — if this test fails, the prefix moved, which is the bug.
"""

from pathlib import Path

import numpy as np

from tichu_training.featurizer import (
    FEATURIZER_OUTPUT_DIM,
    FEATURIZER_VERSION,
    featurize,
)

from .v6_golden_states import golden_states

V6_DIM = 591
_GOLDEN = Path(__file__).parent / "fixtures" / "v6_golden.npz"


def test_v7_feature_vector_is_v6_plus_a_suffix():
    assert FEATURIZER_VERSION == "v7"
    assert FEATURIZER_OUTPUT_DIM > V6_DIM, "v7 must ADD dims, never remove them"

    with np.load(_GOLDEN) as z:
        golden = z["vectors"]
    states = golden_states()
    assert golden.shape == (len(states), V6_DIM), (
        "golden fixture is stale — it was captured for a different state set"
    )

    for i, state in enumerate(states):
        vec = featurize(state)
        assert vec.shape == (FEATURIZER_OUTPUT_DIM,)
        np.testing.assert_array_equal(
            vec[:V6_DIM], golden[i],
            err_msg=(
                f"v7 prefix diverged from v6 on golden state {i}. New sections must "
                f"APPEND to the end of SECTION_DIMS, never insert or reorder."
            ),
        )
