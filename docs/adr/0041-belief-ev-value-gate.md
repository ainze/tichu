---
status: proposed
---

# Price the opponent-modelling program with a Belief-Optimal Chooser, not a perfect-info policy

The opponent-modelling / belief program has never had a number a ship decision can
consume: every result ([ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md),
[ADR-0028](0028-belief-input-compressed-history-projections.md),
[the belief-history gate](../notes/2026-06-03-belief-history-gate.md)) is in
*prediction-accuracy* units. This ADR records the design of the gate that converts it
into **tournament points**, and — more importantly — records that the instrument the
program was *about* to build would have measured the wrong quantity.

## The decision

Measure **Belief EV Value** = `D_on − D_off`: the paired tournament-point gap between two
**Belief-Optimal Chooser**s, both scored against the frozen champion — one drawing
**Determinized Worlds** from the **Belief Model**'s marginals, one drawing them uniformly
over card-counting constraints. Vocabulary in [CONTEXT.md](../../CONTEXT.md)
§"Information-ceiling terms".

The verdict bands are pre-committed:

| `D_on − D_off` | Verdict |
|---|---|
| **≥ +10, CI excluding 0** | GREEN — route to **distillation** of the chooser's decisions (the mine→distill bet), not to a new belief model. The model exists; consumption was the gap. |
| **CI includes 0** | **KILL the card-modelling program.** |
| **significantly < 0** | Not a verdict — a calibration-bug signal. Diagnose, do not report. |

`D_true` (the **True-World Chooser**) and the resulting **Disambiguation Factor** are
reported alongside as diagnostics.

## Why not the instrument we were going to build

The inbound plan was a **perfect-info *policy*** A/B: cotrain an actor on
`featurize_perfect_info` (392 dims) against an identical observable control, and read the
gap as the ceiling of the belief program. Three independent objections retired it.

1. **The RL arm cannot fire a trustworthy KILL.** The actor is KL-leashed to an
   *observable* BC anchor (`kl.play.target: 0.12`), and exploiting hidden hands requires
   diverging from what a player who cannot see them would do — the leash penalises exactly
   the effect being measured. Combined with a lineage that sits −2.69 below the served
   champion after 26,905 iterations, `Δ_oracle ≈ 0` is the expected observation whether
   perfect information is worth 0 points or 200. This is the same structural flaw the
   handoff correctly caught in the *oracle-BC* proposal, relocated from the labels to the
   optimiser.
2. **The true-world ceiling is unreachable, and measurably so.** The
   [blunder-miner](../notes/2026-06-11-blunder-mining-counterfactual-replay.md) found that
   **96–99%** of alternatives that beat the chosen action in the true world **evaporate**
   across observation-consistent worlds. That evaporating share *is* the value of hidden
   information, and no posterior can recover it — acting on the one true world is per-world
   divergence, i.e. **strategy fusion**, the defect that falsified **PIMC**
   ([ADR-0030](0030-phase-2-search-design.md)/[ADR-0031](0031-search-and-learning-loop.md))
   and **piKL** ([ADR-0037](0037-pikl-inference-time-anchored-search.md)). A large
   true-world number would have been a **false GREEN**.
3. **It priced the wrong object.** What a ship decision needs is not "what would perfect
   knowledge be worth" but "what is the belief model we already built worth." Those differ
   by the Disambiguation Factor, which the above measures at 1–4%.

## Design commitments

- **One seat is upgraded, never two.** With every other seat frozen and deterministic, the
  improved seat faces a fixed MDP, so the chooser is one exact step of **policy iteration**
  and `V' ≥ V` is *guaranteed*. Upgrading both team seats is a joint policy change with no
  guarantee. This property is what every prior oracle instrument here lacked — **pMCPA**
  (−67.83, [ADR-0036](0036-pmcpa-runtime-policy-adaptation.md)) and **piKL** (−58.66) both
  went negative because their estimator assumed a continuation the agent then abandoned.
- **Play decisions only.** The Belief Model's entire edge is ADR-0028's *History block*,
  accumulated as the round is played, so `D_on − D_off` is **structurally ≈ 0** at
  round-start decisions. `oracle-call` is retained as a labelled `D_true` diagnostic;
  `oracle-schupfen` is dropped from this gate (it belongs to the partner-coordination /
  JPS pivot).
- **Deadband, not argmax.** `Q` is a K-world mean, and argmax over a noisy estimate is the
  optimizer's curse — the measured mechanism of piKL's −58.66 (32% unstable argmax, 45% of
  decisions overridden on noise). The chooser deviates from the champion only on the
  miner's **validated** tier-2 criterion (K=24 paired worlds, world-win-rate ≥ 0.70, mean
  paired delta ≥ +15), which carries a **measured 0.7% false-positive rate**. Copied
  verbatim rather than re-tuned: re-tuning against the outcome is selection-on-measurement.
- **Everything paired.** Same pool, same seeds, same seat-swap, common random numbers
  across arms. Rounds where both choosers pick identical actions contribute *algebraically*
  zero, not "zero within noise" — the only reason a ~5-point effect is resolvable at all
  against a ±5–7 per-arm CI.
- **The pre-check must be early-stopped, and reported on *both* splits.** Every Round is a
  unique 4-hand deal, so the hidden-hand target is round-unique and an unstopped fit
  **memorises deals**: measured at **train top-1 0.998 / held-out 0.4389 against a floor of
  0.4714**, i.e. worse than card-counting on held-out rounds while near-perfect on train.
  This is the ADR-0033 round-level-split failure mode, and it is why a held-out number alone
  cannot be trusted — the first run's `+0.0011` looked like a clean null and was partly the
  same artifact. `train_selfplay_belief` selects the best held-out epoch; the runner reports
  train and holdout, top-1 and masked BCE.
- **Two pre-checks gate the run.** (a) Belief **calibration on champion self-play** — the
  model was trained on the human BSW corpus and is consumed on self-play, the skew shape
  that produced the `team_scores` and schupfen-`current_player` bugs; self-play carries
  ground-truth hands, so this is nearly free. (b) **Divergence rate** between the belief-on
  and belief-off choosers — piKL's post-mortem named a missing Q-discriminability pre-check
  as its gap. A pilot (~200 deals) sets `n` and reports both before the full run.

## Pre-check result — belief has a small edge in *ranking*, none in *calibration* (2026-07-27)

`scripts/belief_selfplay_precheck.py` over **2,000 champion (`cpfix3328`) self-play Rounds**
= 116,612 Play Decisions, split by whole Round (87k train / 29.6k holdout), early-stopped,
swept over capacity:

| hidden | best epoch | holdout top-1 | Δ vs floor | Δ masked BCE |
|---|---|---|---|---|
| 64 | 8 / 20 | 0.4941 | +0.0227 | −0.0038 |
| 128 | 6 / 20 | 0.4952 | +0.0238 | −0.0018 |
| 256 | 4 / 20 | 0.4953 | +0.0239 | +0.0019 |
| 512 | 2 / 20 | 0.4954 | +0.0240 | +0.0003 |

Converged and capacity-independent across an 8× range, with the best epoch sliding 8→2 as
capacity grows. Two readings:

1. **The pre-check passes, weakly.** A Belief Model *does* generalise past the card-counting
   floor on the champion's own distribution — **+2.4 top-1 points**, concentrated in the
   opening (+0.030) and mid-round (+0.024), vanishing at the end (−0.004) where card counting
   is already near-certain. So `D_on − D_off` is **not** zero by construction.
2. **But calibration is indistinguishable from the floor** (|Δ BCE| ≤ 0.004 everywhere). The
   **Determinization Sampler** consumes *marginals*, not argmaxes — so near-identical
   marginals produce near-identical worlds, and the two arms of the gate may never diverge.
   This is a much weaker prior than the v5 B-core model's ≈ +11 top-1 over its floor on the
   human distribution, and it is why the divergence pre-gate (below) runs before any
   tournament spend.

Positive control: identical code measures **+0.1201** against `RuleAgent` self-play, so the
pipeline detects signal where signal exists. The champion's play is simply far less
informative about its hand — and its floor is far higher (0.471 vs 0.389), because a strong
policy goes out fast and an out opponent has `hand_size == 0`.

**Two false readings were produced and discarded on the way**, both recorded because they
would recur: an unstopped fit at hidden 512 / 12 epochs read `+0.0011` (looked like a clean
null), and at hidden 1024 / 30 epochs read **train top-1 0.998 / holdout 0.4389 against a
floor of 0.4714** — memorising deals while scoring *below* card-counting on held-out Rounds.
Neither is interpretable. Report both splits, always.

### Architecture and parameterisation arms — the ceiling is not a width artifact (2026-07-27)

The width sweep varied the axis this project's head-capacity probe says is the *wrong* one
("schupfen + tichu-call are **depth**-limited; width alone underperforms"), and the fitted
model was found incoherent as a distribution — 0.982 (sd 0.068) of mass per Unseen Card, and
the *publicly known* `hand_sizes` missed by up to 2.4 cards. Four further arms, same cached
116,612 examples, same round-level split, all early-stopped:

| arm | params | holdout top-1 Δ | holder NLL Δ |
|---|---|---|---|
| MLP `hidden=256` (width sweep) | 261k | **+0.0239** | — |
| per-card softmax, `hidden=64` | 53k | +0.0211 | −0.0011 |
| residual `512 × 4` | ~1.4M | +0.0176 | +0.0004 |
| residual `1024 × 4 → 512` (**the play trunk**) | ~4.9M | +0.0157 | **−0.0073** |
| softmax + play trunk | ~4.9M | +0.0123 | +0.0012 |

1. **Depth hurts.** The head-capacity precedent does not transfer — belief is not
   depth-limited; extra capacity only reaches the memorisation slide sooner.
2. **The softmax reparameterisation is neutral.** Making "exactly one holder" true by
   construction bought nothing, because `_draw_holder` already normalises its weights per
   card — the incoherence was never reaching the sampler.
3. **On the metric that governs sampling, every arm is ≈ 0.** `holder_nll` (per-card NLL of
   the true holder, the distribution `_draw_holder` actually draws from — masked BCE scores
   absolute values the sampler never sees) has floor **0.9443** against an uninformative
   `ln 3 = 1.0986`. The best arm improves it by **0.0073 nats, 0.8% relative**.

So `+0.024` is a genuine ceiling across width, depth, architecture and output
parameterisation — not an artifact of the first sweep's axis.

## Divergence pre-gate — the two arms are the same agent 98.6% of the time (2026-07-27)

`scripts/belief_chooser_divergence.py`, 200 champion self-play Rounds, seat 0, the
pre-registered configuration (K=24 paired worlds, k=4, win ≥ 0.70, δ ≥ +15), belief-on =
`belief_h64.bin` (the best-calibrated sweep arm), common random numbers across arms:

| | count | rate |
|---|---|---|
| Decisions the Chooser re-decides | 1,774 | — |
| belief-**off** deviates from champion | 37 | 2.09% |
| belief-**on** deviates from champion | 33 | 1.86% |
| **arms diverge from each other** | **25** | **1.41%** |

`D_on − D_off` can only accumulate where the arms diverge — 0.125 Decisions per Round. Even
granting every divergence the magnitude of a *confirmed robust blunder* (the miner's
re-verified mean, +62) and assuming every one favours the same arm, the difference is bounded
near **≈ 7 pts/Round**; and that bound is absurd on its face, because a divergence is by
construction a near-tie between two candidates evaluated against the same bar. The realistic
magnitude is well under a point.

The pre-registered **GREEN band (≥ +10, CI excluding 0) is therefore structurally
unreachable**, and this is a *mechanical* statement rather than a statistical one: not "we
measured zero within noise" — the noise floor that has swallowed six flat RL results — but
"the two arms are the same agent 98.6% of the time."

Note also that belief-**off** deviates *more* than belief-on (37 vs 33). Belief does not even
find more improvements than card counting; it finds slightly fewer.

**Scope of the verdict.** This is conditional on the pre-registered deadband. A looser bar
would raise divergence — and walk straight back into the optimizer's-curse regime that cost
piKL 58 points/Round, which is precisely why the bar was copied from the miner's validated
0.7%-false-positive criterion rather than tuned. The kill scopes to **cards**: modelling
opponent *policy* is a different object and is not priced here.

**Free by-product, worth more than the verdict.** The belief-off Chooser — one exact step of
policy iteration over card-counting worlds — deviates from the champion at only **2.09%** of
Decisions under a criterion with a measured 0.7% false-positive rate. That bounds the
`mine → distill` headroom on the *same* mechanism, and it is the first EV-unit read on it.

## Consequences

- **The KILL branch is strong in a way no prior belief null was.** The model is consumed
  *optimally* — exact one-step policy improvement, paired worlds, deadbanded, at the only
  decisions where it has information. Any deployed use is strictly worse than this.
- **The kill scopes to *cards*, not to all opponent modelling.** Modelling opponent
  *policy* is a different object and is not priced here.
- **The gate is narrower than the handoff promised.** It answers "what is our belief model
  worth at play decisions" rather than "what is the ceiling of the whole program" —
  breadth traded for a result that resolves to a ship decision.
- **The Belief Model is retrained on champion self-play, not on the BSW corpus.** Both
  on-disk checkpoints are `belief_v5_*` and the featurizer is explicit that *"v5
  checkpoints/bundles are not loadable against v6"* — so there is no usable belief model,
  and the BSW route would need the full re-parse the handoff's Correction 1 flagged. Self-play
  carries ground-truth hands for free, so emitting `(features, labels, mask)` from live
  `GameState`s ([selfplay_emit.py](../../src/tichu_training/belief/selfplay_emit.py)) needs no
  parse pass **and** removes the human-corpus/self-play skew that pre-check (a) exists to
  catch — the ReBeL warning the handoff itself quotes.
- **Featurizer v6 already absorbed ADR-0028's B-core History block** (`declined_top`,
  `lead_summary`, `pass_pressure`, plus a strictly richer `played_by`), so the belief input
  *is* the Feature Vector and no online `HistoryAccumulator` is needed. This also weakens
  ADR-0028's premise for belief: the policy now sees everything belief sees, leaving
  **supervision** as belief's only remaining advantage — a prior-shift against the program
  that this gate exists to price.
- **Nothing new is shippable at inference.** The chooser is ~770 playouts/round — an
  offline instrument, in the same category as the ADR-0030 **Search Agent**. GREEN routes
  to distillation, not deployment.

## Open follow-up — the same channels, in the *policy* rather than a belief net

The rich-history arms above showed the v6 representation **is** lossy: recovering decline
context, combination length, declined Bombs, Trick stakes and the Mahjong-wish void moved
hand prediction from +0.024 to +0.027 top-1 and holder NLL from ≈ 0 to −0.013. Small, but
consistent across all four capacities.

That raises a question this ADR does **not** answer: a Belief Model is an information
*bottleneck* (824 input dims compressed to 168 occupancy probabilities), and ADR-0028 already
warned that feeding its output back "can only re-package information the policy already had".
The policy can consume the same channels **directly**, trained against move quality rather
than hand prediction — which is exactly what Featurizer v6 did to ADR-0028's B-core block.

`scripts/precheck_rich_history.py` prices that on the featurizer's own bar (decile-9
**non-forced** play rows, masked multiclass move prediction, by-game split, early-stopped,
with a row-shuffled **capacity placebo** and per-channel attribution arms). Pre-registered
bar: **≥ +1% relative NLL and clearly above placebo** — the protocol that turned ADR-0039's
apparent +1.16% into a null. **Result pending at time of writing.**

So the verdict here is scoped precisely: **belief as a separate network is not worth
building.** Whether the *history channels* belong in `featurize()` is a live, separately
measured question.
