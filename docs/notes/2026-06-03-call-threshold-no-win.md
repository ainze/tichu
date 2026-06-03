# 2026-06-03 — Call-threshold recalibration: no win; argmax stays

## TL;DR

The behavioral profile flagged master as calling Tichu/Grand ~20% less than
decile-9 humans (an `argmax` artifact). Making it call **more** (lower threshold)
does **not** win — the point estimate trends negative and is almost entirely
call-bonus (−EV marginal calls); CIs cross zero. **Argmax (call iff P(call) ≥
0.5) is fine; the call-conservatism "gap" is not a real weakness.** Confirms the
owner's original intuition that calling "looks fine." The tunable-threshold lever
is shipped (default 0.5 unchanged) but the optimal setting is the default.

## Experiment

Same master checkpoint (full-corpus v5, decile 9), varying only the Call-Network
decision threshold (call iff P(call) ≥ t; 0.5 == argmax). Full-strength
Tournament, 1000 positions × seat-swap (n=2000/pair).
[configs/eval_call_threshold_v5.yaml](../../configs/eval_call_threshold_v5.yaml).

| A vs B | mean Δ (A−B) | call-bonus Δ | 95% CI |
|---|---|---|---|
| t50 (argmax) vs t35 | +4.2 | +4.0 | [−5.6, +15.2] |
| t50 (argmax) vs t25 | +4.6 | +4.5 | [−5.9, +15.2] |
| t35 vs t25 | +1.6 | +1.4 | [−10.4, +13.0] |

## Reads

1. **More calls → worse (or neutral), all via call-bonus.** Lowering the
   threshold adds calls that are net −EV. argmax is the most selective of the
   three and wins the point estimate.
2. **Not significant** — n=2000 vs ±100/±200 call variance gives ~±10-point CIs.
   But there is no hint of improvement; the lever is a non-win.
3. **Mechanism**: the call net imitates "would a human call"; humans over-call
   vs EV, so argmax (>50%-of-humans-would) selects the higher-EV subset. Matching
   the human call *frequency* is not the goal — winning is.

## Bearing on the plan

- The one measured behavioral gap is **not** a weakness. Calls are settled.
- All cheap structural / behavioral levers are now exhausted (BC scale, AWR,
  belief, calls). The surviving lever for play strength remains **PPO sharpening**
  of sub-behavioral decision quality — pending the decision-tape diagnosis to
  confirm the gap is tactical sharpness (vs Resolver / Schupfen).
- Infra kept: `MLAgent(tichu_threshold=, grand_threshold=)`, default 0.5.
