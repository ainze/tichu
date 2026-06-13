"""Distill premise-test mechanics: round-level split, legal-masked loss/metrics, and
the fine-tune loop actually fixing trained corrections while drift is measured."""

import pandas as pd
import torch

from scripts.distill_premise_test import (
    agreement,
    finetune,
    fix_rate,
    masked_logprob,
    rows_from_frame,
    split_by_round,
)


def _frame(n=20, feat_dim=8):
    torch.manual_seed(0)
    rows = []
    for i in range(n):
        legal = [0, 3, 7, 11]
        rows.append({
            "round_idx": i,
            "features": torch.randn(feat_dim).tolist(),
            "legal_idx": legal,
            "chosen_idx": legal[i % 4],
            "alt_idx": legal[(i + 1) % 4],
        })
    return pd.DataFrame(rows)


class _TinyPlay(torch.nn.Module):
    """BCModel-shaped stand-in: features+skill -> dict with a 'play' head."""

    def __init__(self, feat_dim=8, out_dim=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 64), torch.nn.GELU(),
            torch.nn.Linear(64, out_dim),
        )

    def forward(self, features, skill):
        return {"play": self.net(features)}


def test_split_by_round_is_deterministic_and_disjoint():
    df = _frame(40)
    train_a, held_a = split_by_round(df)
    train_b, held_b = split_by_round(df)
    assert set(train_a["round_idx"]) == set(train_b["round_idx"])
    assert set(train_a["round_idx"]).isdisjoint(set(held_a["round_idx"]))
    assert len(held_a) > 0 and len(train_a) > len(held_a)


def test_masked_logprob_normalizes_over_legal_only():
    logits = torch.tensor([0.0, 5.0, 1.0, 1.0, 99.0])  # index 4 is ILLEGAL here
    lp = masked_logprob(logits, legal_idx=[0, 2, 3], target_idx=2)
    probs = torch.softmax(torch.tensor([0.0, 1.0, 1.0]), dim=0)
    assert torch.isclose(lp, probs[1].log(), atol=1e-5)


def test_fix_rate_counts_topk_over_legal():
    model = _TinyPlay()
    rows = rows_from_frame(_frame(12), skill_decile=9)
    rate = fix_rate(model, rows)
    assert 0.0 <= rate <= 1.0


def test_finetune_fixes_train_corrections_and_reports_drift():
    torch.manual_seed(1)
    model = _TinyPlay()
    anchor = _TinyPlay()
    anchor.load_state_dict(model.state_dict())
    df = _frame(24)
    rows = rows_from_frame(df, skill_decile=9)
    before = fix_rate(model, rows)
    finetune(model, anchor, rows, ordinary_rows=rows[:6], epochs=60, lr=5e-2,
             anchor_coef=0.0)
    after = fix_rate(model, rows)
    assert after > before
    assert after > 0.9  # unconstrained fine-tune must nail the train split
    drift = agreement(model, anchor, rows[:6])
    assert 0.0 <= drift <= 1.0


def test_anchor_coef_limits_drift():
    torch.manual_seed(2)
    free = _TinyPlay()
    anchored = _TinyPlay()
    anchored.load_state_dict(free.state_dict())
    anchor = _TinyPlay()
    anchor.load_state_dict(free.state_dict())
    df = _frame(24)
    rows = rows_from_frame(df, skill_decile=9)
    finetune(free, anchor, rows, ordinary_rows=rows[:8], epochs=40, lr=5e-2,
             anchor_coef=0.0)
    finetune(anchored, anchor, rows, ordinary_rows=rows[:8], epochs=40, lr=5e-2,
             anchor_coef=50.0)
    assert agreement(anchored, anchor, rows[:8]) >= agreement(free, anchor, rows[:8])
