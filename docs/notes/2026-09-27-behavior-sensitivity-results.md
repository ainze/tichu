# 2026-09-27 — Behavior Sensitivity Probe: results

**Pre-registration:** [2026-09-27-behavior-sensitivity-preregistration.md](2026-09-27-behavior-sensitivity-preregistration.md) ·
**Data:** [results.csv](2026-09-27-behavior-sensitivity-results/results.csv) ·
[verdict.md](2026-09-27-behavior-sensitivity-results/verdict.md) ·
[deltas.yaml](2026-09-27-behavior-sensitivity-results/deltas.yaml) (frozen δ)

## TL;DR

**Global null.** None of the four behaviors on which the v7 champion
(`cotrain_v7_gated` iter-15360) overshoots top-decile humans is a strength lever.
Nudging each ~6–10 pt in either direction loses EV or does nothing; no arm gains
and every arm's upper CI is below +1.0/Round. Moving *toward* the human rate hurts
or is neutral in all four, so the drift away from humans the
[drift benchmark](2026-09-26-behavioral-drift-v7-champion-vs-bc.md) found is not
critic-driven overshoot. Per the pre-registered rule, the "tune a style metric"
question is closed at v7.

## Setup (as pre-registered)

Treatment = champion + **Behavior Bias** (δ on one **Lever**'s target-action play
logits, in its situation only). Reference = unbiased champion. Both vs the
champion's BC warm-start stack, pool seed 0, 20,000 deals × Seat-Swap per arm,
paired deal-for-deal; deal-cluster bootstrap (10,000 draws); Holm over 8 arms.
δ frozen from 2,000 calibration deals (pool seed 100000) to flip 10 pt of each
Lever's situations. **Identity check passed:** δ = 0 reproduced the reference arm
exactly on 200 deals. **Manipulation check passed** in every arm (|Δrate| ≥ 5.9 pt).

## Results

| Lever | arm | δ | Lever rate | ΔEV/Round [95% CI] | card play | Holm |
|---|---|---|---|---|---|---|
| pass_vs_opponent | + | +3.48 | 25.8 → 35.7% | **−5.55** [−6.74, −4.36] | −1.70 | yes |
| pass_vs_opponent | − | −4.25 | 25.8 → 16.3% | **−2.74** [−3.91, −1.58] | −2.46 | yes |
| phoenix_in_combination | + | +5.15 | 13.4 → 20.6% | −0.45 [−1.01, +0.11] | −1.44 | no |
| phoenix_in_combination | − | −8.00 (cap) | 13.4 → 6.4% | **−3.42** [−4.04, −2.82] | −0.09 | yes |
| dog_lead | + | +1.60 | 54.8 → 63.5% | −0.16 [−0.44, +0.12] | −0.09 | no |
| dog_lead | − | −1.51 | 54.8 → 45.1% | +0.15 [−0.14, +0.44] | +0.27 | no |
| single_lead | + | +3.01 | 51.2 → 57.8% | −0.10 [−1.02, +0.78] | +0.13 | no |
| single_lead | − | −2.74 | 51.2 → 45.3% | **−2.39** [−3.27, −1.52] | −1.70 | yes |

Verdicts: pass_vs_opponent **sharp optimum**; phoenix_in_combination and
single_lead **one-sided optimum**; dog_lead **flat optimum**; **global null: yes**.

## Reading

1. **Contesting Tricks is tuned tightly.** It is the only sharp optimum. Passing more
   (toward humans' 30.7%) costs −5.55, and most of that (−3.85) is in the Call-bonus
   split rather than card play: giving up Tricks changes which Calls land.
2. **The human-ward direction never pays.** Passing more, more Phoenix combinations,
   fewer Dog leads and fewer single leads are all ≤ 0 (the last −2.39).
3. **The Dog lead is free style.** ±10 pt costs nothing either way.
4. **The Phoenix-down arm's loss is all Call bonus** (card play −0.09 of −3.42).

## Limits

- **No placebo arm.** Any perturbation of a trained greedy policy probably costs
  something, so "sharp" partly reflects that. A random-target bias arm would size
  it; it was not pre-registered.
- **The bias is uniform over the Lever's situations** along the policy's own margin.
  It rules out "change how often it does X", not "do X in better spots".
- **Against the BC only.** No arm gained, so the pre-registered champion-opponent
  follow-up never triggered.
- `single_lead`'s Lever situation requires a non-single lead to be legal, so its
  calibration rate (46.2%) sits below the drift cell (51.2%); the realised Δ was
  read on the drift cell.

## Found while building

- **Serial ≠ parallel for ML agents.** An ML agent breaks ties between concrete
  realisations of one Intent by legal-enumeration order, which follows the hand
  frozenset's iteration order, and pickling a Position to a worker rebuilds that
  order. Every parallel run is alike, so every arm ran with `workers > 1` and the
  identity check guards the pairing.
- The detached run's console log is UTF-16 (PowerShell `*>>`); decode it before
  grepping.
