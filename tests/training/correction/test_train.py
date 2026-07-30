"""Preference Correction training mechanics (ADR-0042).

Two things here are load-bearing and everything else is a dial:

  * the train/held-out split is by ROUND. Corrections from the same Round share
    a deal and a policy trajectory, so a row-wise split leaks — and the held-out
    transfer number is what SELECTS the variant that goes to the Tournament.
  * `argmax_is_chosen` must read ~1.0 before training. Every `chosen` in the
    corpus IS the base policy's legal-masked argmax by construction, so anything
    below 1.0 means the corpus was mined against different weights than the ones
    being fine-tuned — a silent wrong-experiment, the trap the session's method
    notes call "verify the artifact before trusting the experiment".
"""

import pandas as pd
import pytest
import torch

from tichu_training.correction.train import corpus_dials, split_by_round


def _rows(specs):
    """specs: (round_idx, is_correction, chosen, alt, legal)."""
    return [
        {"round_idx": r, "is_correction": c, "chosen_idx": ch, "alt_idx": a,
         "legal_idx": legal, "mean_delta": 50.0}
        for r, c, ch, a, legal in specs
    ]


def test_split_by_round_never_leaks_a_round_across_the_boundary():
    df = pd.DataFrame([{"round_idx": r, "n": i}
                       for r in range(40) for i in range(3)])
    train, held = split_by_round(df)
    assert len(train) + len(held) == len(df)
    assert not set(train["round_idx"]) & set(held["round_idx"])
    assert 0.15 < len(held) / len(df) < 0.35
    # a Round's rows travel together, all three of them
    for r in set(held["round_idx"]):
        assert (held["round_idx"] == r).sum() == 3


def test_argmax_is_chosen_reads_one_on_a_matching_checkpoint():
    """The provenance guard. `chosen` is the base policy's argmax over the legal
    set for every row, so a matching checkpoint scores 1.0 and the fix rate is 0
    by construction."""
    rows = _rows([(0, True, 2, 6, [2, 3, 6]), (1, False, 1, 4, [1, 4, 5])])
    z = torch.zeros(2, 8)
    z[0, 2] = 5.0   # chosen is argmax over its legal set
    z[1, 1] = 5.0
    d = corpus_dials(z, rows)
    assert d["argmax_is_chosen"] == pytest.approx(1.0)
    assert d["fix_rate"] == pytest.approx(0.0)
    assert d["preservation"] == pytest.approx(1.0)


def test_argmax_is_chosen_detects_a_mismatched_checkpoint():
    """Fine-tuning weights the corpus was not mined against answers a different
    question entirely, and nothing downstream would notice."""
    rows = _rows([(0, True, 2, 6, [2, 3, 6]), (1, False, 1, 4, [1, 4, 5])])
    z = torch.zeros(2, 8)
    z[0, 3] = 5.0   # argmax is neither chosen nor alt -> wrong weights
    z[1, 4] = 5.0
    assert corpus_dials(z, rows)["argmax_is_chosen"] == pytest.approx(0.0)


def test_dials_separate_the_two_classes():
    """Fix rate is over Verified Corrections only; preservation is over Verified
    Non-Corrections only. Pooling them would let damage on the 31:1 majority hide
    behind transfer on the minority — the failure the June corpus could not even
    see, having discarded the majority."""
    rows = _rows([
        (0, True, 2, 6, [2, 6]),    # correction, FIXED below
        (1, True, 2, 6, [2, 6]),    # correction, not fixed
        (2, False, 1, 4, [1, 4]),   # non-correction, preserved
        (3, False, 1, 4, [1, 4]),   # non-correction, BROKEN below
    ])
    z = torch.zeros(4, 8)
    z[0, 6] = 5.0   # flipped to alt -> fixed
    z[1, 2] = 5.0   # still chosen  -> not fixed
    z[2, 1] = 5.0   # still chosen  -> preserved
    z[3, 4] = 5.0   # flipped to alt -> broken
    d = corpus_dials(z, rows)
    assert d["fix_rate"] == pytest.approx(0.5)
    assert d["preservation"] == pytest.approx(0.5)
    assert d["n_corrections"] == 2 and d["n_non_corrections"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_train_arm_works_on_a_non_cpu_model():
    """The corpus row indices are built on CPU while the model runs on the GPU, so
    the loss's gather/where must see one device. Caught in a live smoke run, not by
    any CPU-only test — hence the explicit CUDA case."""
    import torch.nn as nn

    from tichu_training.correction.train import train_arm

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 8)

        def forward(self, feats, skill):
            return {"play": self.lin(feats)}

    model = Tiny().cuda()
    rows = [{"features": torch.zeros(4), "skill": torch.tensor([0]),
             "legal_idx": [2, 6], "chosen_idx": 2, "alt_idx": 6,
             "is_correction": True, "mean_delta": 50.0, "round_idx": 0}]
    train_arm(model, rows, [], arm="pair", epochs=2, lr=1e-3,
              anchor_coef=0.0, delta_scale=50.0)
