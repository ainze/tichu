# 2026-06-11 — Distill premise test: mined corrections TRANSFER, but naive distillation is EV-negative

## TL;DR

Follow-up to the blunder-mining note (same day). Scaled confirmation to the full
[15,50)-hindsight band: **663 verified corrections** (7.3% of 9,108 candidates at 24
worlds — matching the sampled estimate), then ran the cheap decisive test for the
mine→distill bet (`scripts/distill_premise_test.py`):

1. **Transfer is REAL.** Fine-tuning the play net on 483 corrections (legal-masked CE
   + KL-to-anchor) fixes **20.6% of held-out corrections from disjoint rounds**
   (baseline exactly 0% by construction), at 0.9% top-1 drift on ordinary decisions.
   The blunders share structure a net can generalize — the "rule-less" clustering
   verdict was about *human-legible* keys, not learnability.
2. **Naive distillation still loses.** Head-to-head vs iter_06225 (n=8,000):
   the transfer-maximal variant is a clear **regression −18.2 [−23.0, −14.0]**; a
   strongly-anchored variant (13.9% transfer, 0.5% drift) is **neutral −1.4
   [−6.2, +3.2]**. The arithmetic: fixing ~14–21% of ~0.25 blunders/round buys ≤
   +2–3/round, while every unverified drifted decision fights a policy that 15
   inference-time interventions showed is already calibrated — damage scales faster
   than gain. **Best case neutral; kill naive CE-distillation at this scale.**

## The trade-off curve (two measured points)

| variant | train fix | held-out fix | ordinary drift | h2h vs iter_06225 (n=8k) |
|---|---|---|---|---|
| lr 1e-4, 200 ep, anchor 10 | 98.3% | **20.6%** | 0.9% | **−18.2 [−23.0, −14.0]** |
| lr 5e-5, 100 ep, anchor 100 | 64.0% | 13.9% | 0.5% | −1.4 [−6.2, +3.2] |

Both h2h runs: fine-tuned play net + unchanged iter_06225 schupfen/calls, seat-swap
over the first 4,000 eval-pool deals. Weight recovery from the TorchScript export is
exact (before-fix-rates read 0.000/0.002).

## Why this is informative, not just a null

- The **signal has learnable structure** (held-out transfer ≥ 14% at ≤ 0.9% drift).
  What failed is the *delivery mechanism*: hard CE targets on 483 states perturb a
  calibrated 40M-param policy out-of-distribution faster than they repair it.
- The gain ceiling at current volume (+2–3/round) is below the 4k-deal CI resolution
  (±4.7) — even a damage-free distill could not prove itself without ~10× more
  corrections (≈30k mined rounds, ~30 h compute) *and* a tighter eval.
- The surviving design is the one that delivers the same verified counterfactual
  signal **inside the policy's own trust region**: PPO whose play-decision advantages
  come from paired playouts (vine-style branch-and-compare in the known self-play
  world, luck cancels by construction) instead of `R − V(s)`. Corrections then arrive
  as clipped, KL-bounded gradient weight — not as labels that fight calibration.
  Nothing in this codebase has tested that estimator; the blunder-miner is its
  evaluation-side proof of mechanism.

## Artifacts

`data/runs/blunder_mining_v1/`: `corrections.parquet` (663 confirmed, featurized),
`ordinary.parquet` (2,018 drift rows), `distill_premise_result.json`,
`distill_premise_strong_anchor.json`, `export_ft/policy.pt` (strong-anchor variant),
h2h console logs, `driver_verify_band_emit.py`, `driver_h2h_finetuned.py`.
Rerun: `py -m scripts.distill_premise_test --corrections ... --ordinary ...
--config configs/cotrain_wish_v5.yaml --play-checkpoint <policy.pt|.bin>`.
