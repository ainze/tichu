---
status: proposed
---

# Preference Correction — verified corrections delivered as a pairwise ordering

This reopens **two** levers that carry pre-registered kills: naive CE-distillation
of mined corrections ([2026-06-11 note](../notes/2026-06-11-distill-premise-test.md),
*"best case neutral; kill naive CE-distillation at this scale"*) and the
paired-playout family ([ADR-0040](0040-frozen-reference-vine-pooled-ratchet.md),
*"no vine variant returns without qualitatively new evidence"*). Re-entering a
pre-registered kill silently is how a codebase gets its fourth autopsy, so both
exemptions are stated below with the axis each is claimed on, and the design is
funded only as far as a cheap pre-committed screen.

## What is actually new

**1. The kill note's own diagnosis was incomplete, and the missing half is fixable.**
It attributed the −18.2 to "hard CE targets perturb a calibrated policy out-of-
distribution faster than they repair it". The harness shows two more specific
defects (`data/runs/blunder_mining_v1/driver_verify_band_emit.py`):

- **The loss had no negative class.** The [15,50) band verify screened 9,108
  candidates, confirmed 663, and **discarded the other 8,445 verified negatives**
  — a 12.7:1 majority saying "in this state, what the policy already does is
  correct". Trained on positives only, CE can express nothing but *"prefer Pass /
  prefer the combo more often"* — a soft blanket rule. The
  [2026-06-11 mining note](../notes/2026-06-11-blunder-mining-counterfactual-replay.md)
  §3 had already diagnosed this hazard for hard-coded rules (*"~95% of trigger
  states are the agent being RIGHT — a blanket rule would re-run the
  `forced_yield_opp_caller` disaster (−109)"*) and the distiller then committed
  the same error in gradient form. **Transfer and damage were one phenomenon:**
  20.6% held-out fix rate and −18.2/round are two readings of the same blanket
  rule.
- **The regulariser defended the wrong distribution.** The drift guard was one
  uniformly-sampled decision per Round — **39.1% of them forced** (`n_legal ≤ 1`),
  52.3% with ≤2 legal Intents, median turn 29. The corrections sit at median turn
  46, 5 legal Intents, 0% forced. Roughly 40% of the KL budget anchored no-ops
  while the states where the blanket rule did its damage had no representation in
  the loss at all.

**2. The eval can now resolve the effect.** The kill note's second blocker —
*"even a damage-free distill could not prove itself"* at ±4.7 / 4,000 Starting
Positions — was fixed by the paired seat-swap cluster bootstrap plus pooling
(`d47ee54`, `28feee5`, `99f9516`): **±1.17 at 40k**, ≈ ±2.6 at 8k.

**3. What is NOT new, on the record.** The kill note's *"~10× more corrections"*
blocker is **retired, not satisfied** — it existed only because the ceiling sat
below eval resolution. And the reopening does **not** rest on a larger recoverable
pool. That claim was checked and does not hold; see the arithmetic below.

## Corrected arithmetic (the handoff's numbers were wrong in our favour)

The handoff justified the reopening with *"the blunder rate is 2.3× higher (0.583
vs ~0.25/round), so the gain ceiling scales to +5–7"*. Recomputed on the **same**
band-population estimator for both runs (June's `~0.25` was a crude
`5.3 decisions/round × 3–7%` blend, never the stratified extrapolation):

```
                      [15,50)         [50,150)        [150,400)    rate/round  mean d  pool
iter_06225 (Jun)  11/150 x 9,108   4/150 x 12,358   2/150 x 9,393    0.374      63.1   23.6
iter_27008 (Jul)  15/150 x 4,510   6/150 x  6,334   5/150 x 5,116    0.583      67.1   39.1

candidates/round:   3.04 vs 3.01  | 4.12 vs 4.22  | 3.13 vs 3.41   <- identical
survival 17/450 vs 26/450                       Fisher exact two-sided p = 0.211
```

- The ratio is **1.56×, not 2.3×**.
- Candidate populations per Round are identical to two decimals and survivor mean
  delta is unchanged (63.1 vs 67.1) — the whole difference is a **p = 0.211**
  survival difference. **The recoverable pool has not been shown to have changed.**
  This is consistent with the same session's finding that identifiable loss is flat
  between cpfix3328 (34.6 pts/round) and iter_27008 (39.1).
- But the June kill was decided against an **under-stated ceiling**:
  `0.374 × 0.175 × 63.1 = +4.1/round`, not the "+2–3" that was written down.

~~**Committed planning band: ceiling +4 to +7/round, pool unchanged.**~~
**SUPERSEDED by the exhaustive build — see §Realised corpus: the true ceiling is
+3.8/round.** Both figures above are stratified 150-per-band extrapolations and both
read ~1.6x high. The reopening still rests on the eval fix and the mechanism
diagnosis, not on pool growth; the numbers are left visible because the *estimator*
being biased high is the lesson.

## The two exemptions, and their axes

**ADR-0040 (paired-playout family).** The Blunder Miner and the vine are the same
estimator — force an alternative, play to round end under a frozen field, luck
cancels, so an exemption has to be argued rather than assumed.
ADR-0040 §3 plays **one** world per branch ("exact given the world") and its own
autopsy §2 measured per-state paired-Δ sd ≈ 140 *across* continuations — so
`min_abs_advantage: 5` was a noise filter passing noise, structurally the same
variance wall that killed piKL (per-decision Q ~16 vs per-world SE ~17). The Miner
requires **≥70% agreement across 24 Determinized Worlds** at mean δ ≥ +15, with a
measured 0/150 false-positive control.

**The exemption is invariance, not the world count.** A larger N would be a
quantitative claim about the same statistic, which is exactly what ADR-0040's kill
forecloses. Measured on the iter_27008 screen:

```
survivors (n=26)         identical delta in ALL 24 worlds: 17/26 = 65%
                         win_rate exactly 1.0:             18/26 = 69%
                         spread quantiles: 0  0  0  160  340  500
                         of the 9 WITH spread: median SE@24 = 11.2 (vs the +15 bar)
non-survivors (n=424)    spread == 0 in 0%;  per-world sd 134 -> SE@24 = 27.3
```

Two thirds of Verified Corrections win by an **identical margin in every world** —
the alternative is better however the hidden cards are partitioned. That is a
near-deterministic property, and it is one the vine at N=1 *cannot observe at all*:
a single world cannot distinguish an invariant +60 from a +60 lottery. The filter
is a test for the very property that makes a label trustworthy, which is a
different instrument rather than a longer run of the same one. **The paired-playout
signal stays dead as a per-decision gradient at N=1; it is exempted as a label
selected for cross-world invariance.**

Consequence for §Decision 1's margin: `mean_delta` is *exact* for the invariant
two-thirds and carries SE ≈ 11 for the variable third — so the margin ordering
within that third is substantially noise unless it is re-measured (§Decision 7).

**The self-inconsistency family (pMCPA / piKL / vine / ADR-0031).** A Verified
Correction holds *given all four seats continue with the current policy*; the
update changes that field. This design **is** in that family. The exemption is that
the error is second-order in the size of the update: piKL overrode the anchor on
45% of decisions and pMCPA re-fit every Round, whereas Preference Correction moves
`0.332 × fix_rate ≈ 0.06` of ~35 Decisions per Round — a **~0.17%** behaviour change
(the realised rate makes this exemption *stronger* than the 0.583 estimate did, and
the ceiling correspondingly smaller — the same fact cuts both ways).
That is not a claim to be outside the family; it is a claim to **stay inside the
radius where the labels remain valid**, and §Decisions 4–5 make it a constraint
rather than a hope.

## Decisions

1. **The loss is a sign-verified pairwise ordering, not a distribution target.**
   One row per tier-2 verdict, `(Feature Vector, chosen Intent, alternative Intent,
   verdict)`. Verified Correction → constrain `z[alt] > z[chosen]`, margin scaled by
   verified mean δ. Verified Non-Correction → constrain `z[chosen] ≥ z[alt]`. The
   remaining 1,807 Intent logits are **unconstrained** — the calibration that 15
   inference-time interventions certified is what the loss is silent about, not
   what it overwrites. Rejected: CE toward the alternative (the killed mechanism);
   masked log-softmax margins (couples the whole legal set back in, i.e. a
   distribution target by another name).
2. **No class rebalancing.** Verified Non-Corrections outnumber Verified
   Corrections ~17:1 and that ratio stands. On a negative row the constraint is
   already satisfied at initialisation (`chosen` **is** the argmax), so negatives
   contribute **zero gradient until the positives start dragging them** — a
   barrier, not a pull. Population-proportional weighting is exactly the statement
   "on ~94% of trigger states you are already right"; rebalancing would destroy
   that property and restore the blanket-rule regime.
   **The loss is normalised by the Verified Correction count, not the row count**
   (`tichu_training/correction/loss.py`). A mean over rows would shrink every step
   in proportion to how many Verified Non-Corrections the batch happens to carry,
   so `delta_scale` and `lr` swept on one corpus would not transfer to a corpus
   with a different class ratio — and §Decision 6 deliberately changes that ratio
   by mining all three bands. Under correction-count normalisation a satisfied
   negative is free of **both direction and scale**, which is what makes the
   barrier property exact rather than approximate. A correction-free minibatch is
   routine (~94% of screened states) and is defined by clamping the divisor at 1.
3. **A KL anchor is retained but on a redrawn ordinary set with forced Decisions
   excluded.** Its coefficient is a selection dial, not a design decision.
4. **Single-shot expectation is +2–4, not +7.** The +4–7 ceiling is the *pool*;
   reaching it requires a behaviour change large enough to invalidate its own
   labels. Bars are set to the self-consistency radius, not the pool.
5. **The mechanism is policy iteration.** mine → correct → **re-mine against the
   corrected policy** → correct again. Each shot is small and its labels are
   re-derived under the field that shot created. This yields a convergence signal
   no prior lever had: if the corrected policy's blunder rate falls on a re-mine,
   the loop is real; if the rate holds at 0.583 while h2h claims a gain, the labels
   are chasing their own tail.
6. **Corpus: all three Delta Bands against `iter_27008`,** built by
   `scripts/mine_corrections.py` (logic in `tichu_training/correction/corpus.py`).
   A third entry point beside the two existing miners, because they answer
   different questions: `mine_blunders` screens the extreme tail ("show me the
   worst mistakes"), `mine_blunders_stratified` samples 150/band with a Control
   Band ("what is the blunder *rate*"), and only this one is **exhaustive over the
   bands, keeps the failures, and featurizes** ("what do we train on"). Its tier-1
   sweep is
   already on disk (162,467 rows / 1,500 Rounds), so this is verification only —
   15,960 candidates ≈ 11 h at 1,440/h/10 workers, yielding ~875 Verified
   Corrections and ~15,100 Verified Non-Corrections. Rejected: `[15,50)`-only
   (~3 h). The June corpus was `[15,50)`-only, so the 14–21% transfer figure — the
   entire empirical basis for "the signal is real" — was measured on corrections
   worth 33 points each, while `[50,150)` and `[150,400)` carry 96 and 113 and hold
   ~half the pool. A band-restricted null would be **ambiguous** (bad loss, or
   signal too weak?), and an ambiguous null on a twice-killed lever is worse than
   no experiment. Rejected: re-mining `iter_06225` to replicate the June control
   exactly — it proves a mechanism on a retired lineage and the corpus would have
   to be discarded before anything could ship (corrections are valid only for the
   Checkpoint whose argmax produced `chosen`).
7. **Two-stage verification: screen at 24 worlds, re-measure only the MOVABLE
   survivors at 96.** `needs_reverify` fires on a survivor whose per-world delta
   showed cross-world spread; the 65% with zero spread cannot move under any N and
   are skipped, so this costs **~+8% wall (11.1 h -> 12.0 h)**, not 4x. The second
   reading uses a different rng salt (fresh worlds, an independent measurement) and
   the label comes from it. Positive-class precision goes ~89% -> ~99%; measured on
   the 2026-06 re-verify, 49/55 survivors held the same bar at 96 worlds and the
   delta de-inflated 69.6 -> 62.1 (-11%), which matters because the margin IS the
   loss's scale. Both readings stay on the row (`screen_*` + final), because the
   label is a claim about the second and the first is the selection that produced
   it. **A survivor that regresses becomes a Verified Non-Correction — it changes
   class rather than disappearing**, which is only possible because §Decision 1
   keeps both classes; under the June positives-only design it simply vanished.
   Rejected: re-verifying non-survivors. This protocol buys down false POSITIVES
   (which teach a worse action). The negative class is ~1.8% mislabelled (~29% of
   the correction count), but a mislabelled negative is **inert** — satisfied at
   initialisation, zero gradient, and it only pushes back if the corrections drag
   that state, which is precisely when a second opinion is wanted anyway. Rejected:
   raising the screen to 96 for everything (4x wall to re-measure a two-thirds
   majority that cannot move).
8. **Prerequisite bug fix.** `scripts/mine_blunders.py:71` seeds tier 2 with
   `hash((round_idx, turn, alt_str))`; `alt` is a `str`, so the seed is salted by
   `PYTHONHASHSEED` — unset in this repo, and freshly randomised in every `spawn`
   worker. **Tier-2 verification is not reproducible, not even within one run.**
   Hash the int action index (or `hashlib`) before mining anything this design
   depends on.

## Pre-registered bars (committed before any number)

- **Invalid dials, on the record:** held-out fix rate, negative-preservation rate,
  ordinary drift, cluster reports. ADR-0040 already convicted the first
  (*"correction fix-rate (rose while strength collapsed)"*). They are **variant
  selection only** and never a verdict.
- **One variant reaches head-to-head.** Sweep margin/weighting/lr/epochs on the
  offline dials as much as desired — it consumes no evaluation — then commit to a
  single variant. Taking N variants to a tournament and shipping the best is
  selection on noise (the piKL optimizer's-curse failure); the June kill was clean
  precisely because it ran two pre-committed variants rather than a sweep.
- **The CE control arm is mandatory.** Both losses run on the *same* Correction
  Corpus from the *same* base. If CE reproduces ≈ −18 / −1 and PAIR is positive,
  the mechanism claim is demonstrated. If CE also comes out fine, the June kill was
  about lineage or corpus size and the pairwise story is unsupported — worth
  knowing before spending anything further.
- **One Tournament, all pairs measured directly:** `{iter_27008, iter_27008+CE,
  iter_27008+PAIR, cpfix3328}`. Never difference two other pairs (~2.3-point
  non-transitivity observed twice). `iter_27008 vs cpfix3328` at +1.39 [−0.75,
  +3.58] is the harness consistency check.
- **Mechanism works:** PAIR beats `iter_27008`, n=8,192, 95% CI excluding 0.
- **Ship:** PAIR beats **cpfix3328** at n=40k on fresh Starting Positions, 95% CI
  excluding 0, AND Move-Prediction Eval within −3pp (the eval double-bind).
- **Kill, permanently:** PAIR's 8k CI entirely below 0 → the delivery-mechanism
  question is closed for good; fall back to ship `iter_27008` / a different BC /
  critic quality. No third reopening without a qualitatively new label source.
- **Self-consistency wall (the diagnostic kill):** if strength appears only at
  update sizes large enough to break the Verified Non-Corrections, that is the
  wall of §Exemptions 2 and the lever closes for the same reason pMCPA and piKL
  did — visible before anything ships.

## Consequences

- Total cost to a verdict: ~11 h of tier-2 verification (tier 1 already on disk),
  minutes of training per arm, and one 4-agent Tournament. **No new mining is
  authorised until PAIR has cleared the CE control arm.** The handoff had budgeted
  16 h of mining *before* the mechanism was resolved at all.
- A fourth training stage joins BC / AWR Refine / PPO Refine / Full-Stack
  Co-Training. Its output is byte-compatible, so Difficulty-tier swaps stay
  drop-in.
- The Correction Corpus becomes a **versioned, policy-relative** artifact: it is
  invalid for any Checkpoint other than the one it was mined against. This is the
  main ongoing maintenance cost and the reason §Decision 5 is a loop, not a step.
- If killed at the ladder's end, the strength program's remaining paths are ship
  `iter_27008` and accept the plateau, a different BC (the wishfix BC starts −16.29
  / −10.58 behind and co-training closes about half in 27k iterations), or critic
  quality (`project_advantage_snr_probe.md`). That closure statement is deliberate
  and part of the bar.

## Realised corpus (2026-07-30) — the ceiling came down; one bar must move

`scripts/mine_corrections.py` over `iter_27008`, 15,960 in-band candidates, 1,500
Rounds, 7.9 h wall (throughput 2,015 cand/h, better than the 1,440 assumed).
37 candidates (0.2%) skipped for a legal set outside the Action Space.

```
band        screened  corrections  survival  mean d  pool/round  %compute  %pool
[15,50)        4,502          239     5.3%    33.4      5.32       28%      25%
[50,150)       6,322          218     3.4%    85.1     12.37       40%      58%
[150,400)      5,099           41     0.8%   140.1      3.83       32%      18%
TOTAL         15,923          498            64.8     21.5
```

**1. The pool is 21.5 pts/round, not 39.1 — ceiling +3.8, not +4 to +7.**
Exhaustive rate **0.332 corrections/Round** (0.389 pre-re-verify) against the
stratified extrapolation's 0.583. The 150-per-band sample read 26/450 = 5.8%
survival where the exhaustive screen gives 3.66%; 1.6x high, marginally outside its
own binomial CI. **The exhaustive number is authoritative.** Note the corollary:
cpfix3328's 0.324/Round is itself a 450-sample estimate carrying the same upward
bias, so it is *not* directly comparable to 0.332 — the "pool unchanged" conclusion
stands on the identical per-Round candidate populations and p=0.211, not on these
two numbers matching.

**2. The re-verify was the best 8% spent. 40% of movable survivors regressed.**
213 of 583 screen survivors were movable (37%, vs 35% predicted); **85 regressed to
Verified Non-Correction** — 15% of the whole screen-only corpus would have been
false positives, the expensive error class under a pairwise loss. Reconciles with
June's 6/55 = 11% of *all* survivors: at ~70% invariant that is ~36% of movable
against 40% here, the same regime once conditioned on movability. Delta de-inflated
77.9 -> 65.1 (−16%) overall, −4% among those that held.

**3. Invariance is 74% at scale**, above the 65% the screen predicted — the
§Exemptions argument is stronger than it was written.

**4. §Decision 6's all-bands call was right, for the wrong reason.** June's
[15,50)-only corpus captures just **25%** of the pool, so all-bands was correct. But
the ADR justified it via the high band, and `[150,400)` is the *worst* value on the
board: 32% of compute for 18% of pool (0.75 pool-per-1k-screened vs `[50,150)`'s
1.96). **`[50,150)` is the productive band** — 58% of the pool, and absent from
every prior corpus. Next iteration: keep `[15,50)` and `[50,150)`, subsample
`[150,400)`.

**5. AMENDED BAR — the 8k mechanism screen would now manufacture a false null.**
§Decision 4 set single-shot expectation at +2–4 against a +4–7 ceiling; rescaled to
a +3.8 ceiling that is roughly **+1.5–2.5**. The pre-registered screen was "beats
`iter_27008` at n=8,192, 95% CI excluding 0" and 8k resolves ±2.6 — so a true +2
would read CI_lo < 0 and be recorded as a kill. That is precisely the arithmetic
ADR-0040 §4 diagnosed (a gate needing +4.8 while real steps were +0.5–2), about to
be repeated in a different instrument. **Revised allocation, committed before any
arm runs:**

- **CE control arm at n=8,192.** Its predicted effect is large (−18.2 or −1.4);
  8k resolves it and spending more is waste.
- **PAIR at n=40,960** vs `iter_27008` *and* vs `cpfix3328`, both measured
  directly. ±1.17 makes +1.5–2.5 a 1.3–2.1σ read — still not comfortable, which is
  itself the honest finding: **a single shot of Preference Correction is near the
  floor of what this eval can certify.** §Decision 5's policy iteration is not an
  optional refinement; it is how the effect gets large enough to bank.
- A null at 40k is a kill. A null at 8k is **not** — do not record one.
