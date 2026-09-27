# 2026-09-27 — Behavior Sensitivity Probe: pre-registration

**Written before any probe number is seen.** Levers, perturbation sizes, sample
size, bar and closing rule are committed here first. The calibration pass (step 1)
only sets δ; it measures no EV.

## Question

The [Behavioral Drift Benchmark](2026-09-26-behavioral-drift-v7-champion-vs-bc.md)
says *how* the v7 champion (`cotrain_v7_gated` iter-15360) plays. It does not say
which of those behaviors, if changed, would change strength. Correlating Round EV
with the metrics would mostly measure deal quality (a strong hand contests more,
leads more singles *and* wins), so this probe **intervenes** instead: nudge one
behavior at a time and measure the EV change on the same deals.

Hypothesis under test: **for each lever below, the champion sits at a local EV
optimum of that behavior's rate** (a converged policy should, by the envelope
argument). A lever whose nudge gains EV is either unconverged training or critic
bias pushing the policy past its optimum
([advantage-SNR probe](2026-07-24-advantage-snr-probe-critic-bias.md)).

## Levers

Chosen where the champion **overshoots top-decile humans** — the drifts most
likely to be critic-driven rather than discovered. Calls (closed
[2026-07-29](2026-07-29-calls-are-not-a-lever.md)) and Partner Overtake (probed at
v7: forced-yield −11.7/Round) are excluded.

| lever | situation (non-forced Play Decision) | target actions | champion | top humans |
|---|---|---|---|---|
| `pass_vs_opponent` | an opponent holds the Trick, a beat is legal | Pass | 25.8% | 30.7% |
| `phoenix_in_combination` | a multi-card Combination containing the Phoenix is legal | those Combinations | 40.6%¹ | 53.4%¹ |
| `dog_lead` | leading, Dog in hand, partner has not called | the Dog | 54.8% | 41.9% |
| `single_lead` | leading | Singles other than the Dog | 51.2% | 46.8% |

¹ the drift metric (share of Phoenix plays that are inside a Combination). The
probe also reports the lever's own rate, P(Phoenix Combination | one is legal),
which is what the bias moves directly (new recorder field `phoenix_combo_legal`).

**Mechanism — Behavior Bias.** `BiasedMLAgent` (`ml_biased`) is the champion with
δ added to the play logits of every legal target action, in the lever's situation
only, before the greedy argmax. It flips exactly the Decisions where the best
target and the best non-target action are within δ of each other — the marginal
cases, which is the right perturbation for a first-order slope. Every other
Decision (and every Schupfen / Call / Wish / Dragon Decision) is the champion's.

## Design

- **Arms.** Baseline = champion. Treatment = champion + bias (lever, δ). 4 levers ×
  {+δ, −δ} = **8 treatment arms**. Each team plays the champion's **BC**
  (warm-start stack, as in the drift benchmark) over the same Pool deals with
  Seat-Swap — the drift benchmark's Fixed-Opponent design with the unbiased
  champion as the reference arm. The baseline arm is played once and shared.
- **Pairing.** All agents are greedy, so a Round in which the bias never flips a
  Decision is *identical* in both arms (Δ = 0 exactly). Variance comes only from
  Rounds that diverge.
- **Pool.** `pool_seed 0`, **20,000 deals** per arm (40,000 Rounds). Skill Decile 9.
- **δ calibration (step 1, no EV read).** On 2,000 deals of *baseline* play, log
  for each lever situation the gap `g` = best target logit − best non-target logit.
  To first order, +δ flips the situations with −δ < g < 0 to the target, −δ the
  ones with 0 < g < δ away from it. Per lever and sign, δ is the smallest |δ| whose
  predicted flip share is **10 pt** of the lever situations, capped at 8 logits. δ
  values are frozen into the run config before step 3.
- **Identity check (step 2).** δ = 0 on 200 deals must reproduce the baseline log
  byte-for-byte; otherwise stop.

## Outcomes

- **Primary:** ΔEV = mean Subject-minus-opponent points per Round, treatment minus
  baseline, deal-cluster bootstrap 95% CI (`drift_stats`, 2,000 draws), per arm.
- **Manipulation check:** realised Δ of the lever rate. An arm whose rate moves
  < 5 pt is a **failed manipulation**, not a null.
- **Descriptive:** slope = ΔEV / Δrate (points per Round per rate point), the
  drift panel's other metrics (what else moved), card-play vs call-bonus split.

## Decision rules

Multiplicity: Holm–Bonferroni over the 8 arms at family α = 0.05.

1. **Lever found** — an arm with ΔEV > 0 surviving Holm. Follow-ups, in order:
   (a) the same arm against **champion** opponents (is it a BC-exploit?);
   (b) a 40k-deal ship A/B of the biased agent vs the served champion —
   the bias itself is a shippable inference-time tweak; (c) if the winning
   direction is *toward* human rates, record it as evidence of critic-driven
   overshoot and a target for the critic work, not for reward shaping.
2. **At optimum** — no arm of a lever gains. Both arms negative and surviving
   Holm = a **sharp** optimum; neither surviving = **flat** (the calls pattern).
3. **Global null** — no arm gains and every arm's upper CI < +2.0/Round: aggregate
   behavioral style is **not** a strength lever at v7; strength lives in the
   state-specific allocation of these actions. Close the "tune a style metric"
   question; don't reopen it without a new mechanism.

Expected SE, for reading the bar: the drift benchmark's champion-vs-BC margin had
a CI half-width of ~1.6/Round at 20k deals with *every* Round differing; the
paired treatment arms differ on fewer Rounds, so their CIs should be tighter. If
an arm lands with |ΔEV| > 1.0 and a CI straddling 0, it may be extended once to
40k deals (pool seed 20000 for the extra half) — pre-registered here, used at
most once per arm.

## Not in scope

No reward shaping, no retraining, no claims about human opponents (the BC is the
human proxy; see the drift note's caveats). Opponent-model levers, Schupfen, Wish
and Dragon levers are a later round if this one finds anything.
