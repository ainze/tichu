# 2026-08-01 — Legal-move mask as *trunk input*: pre-check GREENLIGHT

Not to be confused with [2026-06-07 `legal_actions` speedup](2026-06-07-legal-actions-memo.md),
which is about the *cost* of enumerating legal actions. This note is about
feeding the resulting mask to the network as an **input feature**.

## TL;DR

The legal-Intent mask is computed at every Decision and applied **only as an
output mask**. It never reaches the trunk. Feeding it in is worth
**+3.04% held-out NLL and +0.81 pt top-1** on non-forced Play Decisions, against
a shuffled-mask capacity placebo that comes out **negative (−2.24%)**, and the
lift is **flat across an 8× rows × 8× width sweep**.

It ships as a **model-arch config flag**, not a `FEATURIZER_VERSION` bump — the
mask is already in every training batch. No re-materialisation, no bundle
invalidation, no inference cost.

Strength is **not** yet demonstrated. This is a BC-fit screen; only a
cotrain/eval A/B settles EV (cf. [calls are not a lever](2026-07-29-calls-are-not-a-lever.md)).

## Where the mask currently lives

| site | uses mask | how |
| --- | --- | --- |
| `bc.loss.masked_cross_entropy` | yes | output mask, illegal logits → −1e9 |
| `ppo.policy.sample_masked` | yes | output mask before sampling |
| `BCModel.forward` | **no** | concatenates only `features + skill_emb` |
| `ValueBaseline.forward` | **no** | no mask at all, not even at the output |

## Method

`scripts/precheck_legal_moves.py`. 300,010 decile-9 Play Decisions replayed from
the BSW archive over 2,660 games. Objective is the **real** BC objective —
masked cross-entropy over the 1809-wide play head against the human's Intent,
with the true legal mask applied at the output **in every arm**. So the delta
measures purely the representational shortcut, not the mask's decision-relevant
content, which every arm already has.

By-**game** three-way split (round-unique leakage would inflate the delta — the
ADR-0033 lesson), seed-averaged, early-stopped on val (the ADR-0039 lesson:
never trust a residual verdict without early stopping). Reported on the
**non-forced** subset (|legal| > 1); forced rows are 39.1% of Play Decisions and
are free for every arm.

## Results (non-forced rows)

| arm | dim | NLL | top-1 | rel NLL |
| --- | ---: | ---: | ---: | ---: |
| A baseline v6 | 591 | 1.1057 | 0.5991 | — |
| **B raw mask** | 2400 | **1.0721** | **0.6072** | **+3.04%** |
| B′ shuffled placebo | 2400 | 1.1305 | 0.5932 | −2.24% |
| C summary-30 | 621 | 1.0907 | 0.6030 | +1.35% |
| D n_legal / is_forced | 594 | 1.1064 | 0.5996 | −0.07% |

The **placebo going negative** is what makes this credible: 1809 noise dims
actively *hurt*, so B is 5.3 points clear of the capacity confound. For
contrast, the v7 trick-stakes pre-check was rejected at sub-1%.

### Value head

`ValueBaseline` has no output mask at all, so it is the one place the mask could
matter structurally — and the critic residual is the standing bottleneck
([advantage-SNR probe](2026-07-24-advantage-snr-probe-critic-bias.md)).

| arm | R² | Δ vs baseline |
| --- | ---: | ---: |
| A baseline v6 | 0.2753 | — |
| B raw mask | 0.2967 | **+0.0214** |
| B′ shuffled placebo | 0.2794 | +0.0041 |
| C summary-30 | 0.2949 | +0.0196 |

≈ **+0.017 R² clear of placebo**.

### Scale sweep — the decisive check

A hand-computed shortcut helps most when the model is data-starved, so the lift
*should* decay as width and rows grow if it is an artifact. It does not:

| rows | hidden 128 | hidden 512 | hidden 1024 |
| ---: | ---: | ---: | ---: |
| 37,156 | +2.80% | +2.09% | +1.43% |
| 74,174 | +3.87% | +3.67% | +3.69% |
| 147,116 | +3.68% | +3.80% | +3.36% |
| 300,010 | +3.22% | +3.01% | **+3.54%** |

Flat across 8× data and 8× width; the largest cell is near the top of the table.
This removes the main reason to expect it to vanish at production scale — it
does not *prove* survival at 1.25B rows against a 1024×4 residual trunk.

## Why it can help despite adding zero information

Legality is a **deterministic function of fields already encoded at full
precision**: `own_hand[56]` is exact, plus `trick_top_combo` / `mahjong_wish` /
`phase`. ADR-0017 Finding 3 establishes that the lossy parts of
`trick_top_combo` (dropped SF-bomb suit, collapsed phoenix position) are not
beat-relevant. So the mask is a pure **computational shortcut**.

Crucially, because output masking is already applied, any rule of the form
"argmax over legals of a per-action score" is learnable **without** the mask as
input — the net emits a score per action and the output mask restricts it. The
mask can only pay where an action's value depends on **what else** is legal.

This is not a footnote; it invalidated two draft positive controls before the
harness was trustworthy. A valid control must be *context-dependent*. The one
used: label = lowest legal index when |legal| is even, highest when odd. Parity
of |legal| is invisible to a per-action scorer, so baseline caps near 50% while
mask arms approach 100%. The harness only became trustworthy once that control
separated (+1.2 to +1.6% for mask arms, −0.69% for the placebo).

## What this refutes

- **"Nothing to gain / it's all capacity."** The negative placebo kills that.
- **"It's the forced-decision story."** `n_legal` / `is_forced` alone is worth
  **zero** (−0.07%). Not the [39%-forced](../../src/tichu_training/ppo/rollout.py)
  mechanism.
- **"A compact summary is the right form"** (the ADR-0017 analogy). The 30-dim
  summary recovers under half the lift → the signal is fine-grained action-set
  structure, and the raw mask is the right form here.

## Honest gaps

- The in-script `redundancy_check` is **vacuous**: 0 duplicate feature vectors
  in 300k rows, so it tested nothing. The derivability claim above rests on the
  ADR-0017 argument alone, not on that check.
- Probe is a 2-layer MLP on 300k decile-9 rows; production is 1024×4 residual on
  1.25B. The sweep bounds the extrapolation risk but does not eliminate it.
- BC-fit lift ≠ strength.

## Next step

The mask is already bit-packed per row in the materialised bundle and already
unpacked into every batch as `tensors["legal_mask"]`, in scope one line above
the `model(...)` call at `bc/training.py:161` and `:357`. So:

1. `BCModel.forward(features, skill_decile, legal_mask)`, `feature_dim + 1809`,
   behind a config knob defaulting **off**.
2. Retrain BC on the existing `materialised_full_v6_wishfix` bundle. The corpus
   is unchanged, so `bc_full_corpus_v6_wishfix_memmap` stays the clean baseline.
3. Greedy gate for the strength verdict.

Keep **out** of that retrain to avoid confounding: dropping forced Play rows
from the BC corpus (they are exactly zero-loss and zero-gradient under
`masked_cross_entropy` too — verified loss `-0.000e+00`, grad-norm `0.000e+00` —
so the PR #77 rollout argument transfers, but it changes the corpus and needs
its own baseline), and feeding the mask to `ValueBaseline` (different network,
own training path).
