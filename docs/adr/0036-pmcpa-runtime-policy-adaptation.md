---
status: closed
---

# pMCPA: per-round run-time policy adaptation on the paired-advantage estimator

> **CLOSED 2026-06-13 — faithful-mechanism kill.** Built and tested end-to-end.
> Three configs, all decisive regressions vs the non-adapted θ_o control:
> uncontained lr=0.05 **−247**; contained lr=0.01, K=128 **−59 [−115, −6]**; and the
> decider — **oracle adaptation on the TRUE world (danger-(b) = 0, the ceiling of any
> world-quality fix) −68 [−94, −39]** (n=200). Because even a *perfect* per-deal
> gradient degrades play, world-sampling bias is NOT the gap and no (b)-rescue
> (belief-on / mid-round re-adapt / bigger K) can help — the mechanism itself is
> broken. Root cause: the paired advantage values an action assuming a **θ_o
> continuation**, but the adapted θ_a then deviates, so the improvement target is
> self-inconsistent (same imperfect-info failure family as ADR-0031 search+learning
> and ADR-0035 re-anchored vine; vine survived only by staying in a tiny
> trust region distributed across many deals). The lever joins the closed ledger;
> the strength program remains where ADR-0035 left it (human-measurement loop). The
> companion **piKL** (inference-time search, no weight change) is the untested
> sibling from the same deep-research run, should the program reopen. Full arc in
> the diagnostic section below. Code (`ppo/pmcpa.py`, scripts, oracle path) retained
> and tested — reusable if a fundamentally different estimator ever justifies it.

[Suphx](https://ar5iv.labs.arxiv.org/html/2003.13590)'s pMCPA (parametric Monte-Carlo
Policy Adaptation) fine-tunes the policy *at inference time*, per round, on rollouts from
worlds consistent with the actor's own Hand, then resets — belief is baked into the
sampling distribution, no tree search, no public-belief state. It is the natural fit for
Tichu's irregular play order (bombs / Dog / wish), and the **one self-play lever not yet
on the closed ledger**: it conditions the gradient on the *actual deal* and **applies the
correction to the round it was computed on instead of distilling it across rounds** —
precisely the move the vine autopsy ("enrich, don't extend") never tried.

This ADR records the design as walked in the grilling pass of 2026-06-13 (handoff
`handoff-pmcpa.md`, dated 2026-06-11). It reopens, *as a probe not a program*, the
strength line that [ADR-0031](0031-search-and-learning-loop.md) and
[ADR-0035](0035-vine-paired-advantage-ppo.md) closed. **No code written yet.**

## Why this is not just another closed lever

Three documented failures bound the design, and pMCPA threads between them:

- **The Play-phase premise is refuted** ([ADR-0031](0031-search-and-learning-loop.md)):
  forced-press −1.4 (neutral), forced-bomb −17.4 (harmful) — the master plays the
  diagnosed spots correctly. The located signal is the blunder-miner's
  **~0.2–0.3 blunders/round, ~+60 pts each** (`project_blunder_miner`): real, rare,
  below PPO's noise floor.
- **`R − V` drowns it** ([ADR-0034](0034-full-stack-cotraining-ppo.md) closure): round
  outcome variance ~55–60% even with the perfect-info critic. Suphx's `R(τ)·ratio`
  estimator is *worse* (no baseline) — that is why Suphx needed K=100,000.
- **Distilling paired corrections is flat-then-fatal**
  ([ADR-0035](0035-vine-paired-advantage-ppo.md)): fixed-anchor vine EV-neutral-but-flat
  (+0.45, a 2.1% fix-rate *delivery* failure); re-anchoring −23.59 via **self-play
  co-drift**.

pMCPA inherits the **right** side of each: it uses the **vine paired-advantage
estimator** (luck-cancelled within a world) so it is not Suphx-variance-bound; its field
is **frozen θ_o, never updated across rounds**, so the −23 co-drift is *structurally
impossible*; and it **never generalizes across rounds** — sidestepping vine's delivery
failure — because the correction is applied locally and discarded.

## Decisions (walked one-by-one, 2026-06-13)

**A. Run it as an extensive, faithful test — not a cheap probe.** Every cheap proxy in
this program (forced-press, forced-bomb, value-ceiling) returned "refuted" with a
caveat that it tested a *crude proxy*. pMCPA is run as the real mechanism at growing
power (Decision F), not a 50-round direction-finder.

**B. Estimator = the vine paired-advantage, not Suphx's `R(τ)·ratio`.** Within-world,
luck-cancelled (`A(s,chosen) = R_det(chosen) − mean(R_det over branches)`), reusing
`ppo/vine.py` (`collect_vine_rows`, `playout_from`, `candidate_alternatives`). Faithful
to Suphx's *idea* (per-round adapt-then-reset), not its high-variance estimator — which
would re-inherit the K=100k requirement we cannot afford.

**C. Granularity = per-round-at-start, with a single mid-round re-adapt hook built but
defaulted off.** Per-round is pMCPA's natural unit (amortizes K rollouts over the round)
and is *piKL's* complement, not its overlap (piKL owns per-decision/Schupfen). Two
distinct dangers hide in the handoff's single "small-K overfit": **(a)** gradient
variance — killed by the paired estimator; **(b)** *world-sampling bias* — the K sampled
worlds are a biased draw, θ_a is applied to the one true world not in the sample, and
the paired estimator does nothing for it. **(b) is what the K-sweep actually probes.**
Round-start worlds are maximally diffuse (3 opponents × 14 unknown cards), so the mid-round
re-adapt (re-sample against the tighter public state after the first trick) is the
pre-registered (b)-rescue, not a luxury.

**D. Adapt the play head only (trunk frozen) by default; trunk+play is the escalation.**
Head-only is more robust to the biased K-world sample (danger (b) dominates) and leaves
the shared Trunk that **Wish / Dragon Assignment** ride untouched. Trunk+play
([ADR-0031](0031-search-and-learning-loop.md) Decision H's "head-only too weak") is the
pre-registered capacity escalation for a flat result — the clean "capacity vs premise"
disambiguation. The **field is frozen θ_o on all four seats** during rollouts (the
estimator-honest fixed-field regime that made v1 vine safe). In the A/B both team-0 seats
adapt independently on their own info set; the control is θ_o everywhere; seat-swap
cancels team assignment. Wish/Dragon move-prediction is logged within adapted rounds as a
guardrail.

**E. World fidelity belief-off; in-round containment via clip-0.1 + adv-norm +
KL-anchor-to-θ_o.** Belief-off matches every program measurement; **belief-on is a second
(b)-rescue** (weights worlds toward likely holdings) armed for the decisive run. Void-
honoring in the sampler is confirmed/extended *before* the mid-round re-adapt is flipped
on (no voids at round-start). The per-round **KL-anchor to θ_o** is the guard against
"adapted gets WORSE": it caps how far a biased-sample gradient drags θ_a within the round.
No *cross*-round anchor — θ_a is discarded. Steps: probe = 3; decisive sweeps {1, 3, 5}.

**F. Bar, effect size, compute envelope.** Bar in Tournament-Matrix units, not Suphx's
66% (a Mahjong individual win-rate that does not translate to 2v2 point-deltas): **two
reads**, the mechanism A/B (adapted-θ_a vs non-adapted-θ_o, seat-swapped) and the **ship
bar vs shipped iter_06225, 95% CI > 0** ([ADR-0034](0034-full-stack-cotraining-ppo.md)/
[ADR-0035](0035-vine-paired-advantage-ppo.md) lineage). Realistic target = the
blunder-miner's deliverable, **+2–3/round**. Power-stage the *real mechanism* (not a
proxy): **48 h probe** (K=128, round-start only, n≈8.6k, CI ≈ ±4) → **2-week decisive
run** (n=30–40k, K-sweep {64,128,256}, mid-round re-adapt ON, belief-on armed, CI ≈ ±2).
The paired estimator makes worlds *playout-expensive* (~13 playout-equiv/world), so the
feasible K is **~64–256**, an order of magnitude below the handoff's {500, 2k, 10k}; the
budget math is calibrated off vine's measured ~166 playouts/s.

**G. Wiring.** A new `PMCPAAgent(Agent)` wrapping an `MLAgent`, holding frozen θ_o nets,
a per-round play-head clone (θ_a), an optimizer, and the sampler; round start fires
adaptation via an **optional `on_round_start` hook on `play_full_round`** (additive, off
by default — the existing eval path stays byte-identical). Adaptation playouts run
serially inside one worker; parallelism stays **across tournament rounds** (no nested
spawn pools); torch threads pinned to 1. TDD the pure pieces (world-sampling consistency,
paired-row generation rooted at a real round, θ_a→θ_o reset bit-identical).

## Pre-registered outcome bands (decided before the run)

- **Probe clearly positive** (point est ≳ +3, most mass > 0 at ±4) → fund the 2-week
  decisive run with the K-sweep and rescues armed.
- **Probe flat** (CI spans 0) → does *not* kill (the anti-false-negative stance). Trigger,
  within the decisive budget, the rescues **in order**: (b)-rescue (mid-round re-adapt +
  belief-on) → capacity escalation (trunk+play). **Only a fully-powered, fully-rescued
  decisive run that is still flat is the clean kill** — a *faithful-mechanism* null that
  independently confirms [ADR-0031](0031-search-and-learning-loop.md)'s premise refutation.
- **Probe negative** (adapted clearly worse) → small-K overfit / (b)-bias fatal at this K;
  `diagnose` (K-overfit vs step-size vs sampler-bias) before further spend.

## Strategic placement

- **The product is the *online agent first*, not a training signal.** A large stable edge
  does **not** imply distill-back works — distilling paired corrections is exactly what
  vine did (flat, then −23). The gain may be intrinsically local. A positive pMCPA ships
  as a new inference-time agent (a Search-Agent-class candidate / Difficulty tier);
  distill-back is a separate, later, non-assumed question.
- **It converges with the human-measurement loop, doesn't compete.** A Tournament A/B is
  self-relative and cannot confirm the superhuman goal; a win **routes into** the
  owner-vs-agent human loop (the path chosen at the 2026-06-13 vine close) for
  adjudication.
- **piKL is untouched** — pMCPA owns Play/per-round, piKL owns Schupfen/per-decision.

## Risks

1. **World-sampling bias (b) at the feasible K (~64–256).** The dominant risk; the
   mid-round re-adapt and belief-on are the pre-registered rescues.
2. **Premise is genuinely gone** ([ADR-0031](0031-search-and-learning-loop.md)). The
   faithful-mechanism kill (Band 2) is the honest way to find this out at full power
   rather than via a proxy.
3. **In-round overfit cratering the round.** The KL-anchor-to-θ_o + clip-0.1 + few steps
   are the containment; a negative probe means they were insufficient.
4. **Compute.** Feasible K is an order of magnitude below the handoff's; the bar is set in
   resolvable units (+2–3 at n=30–40k) accordingly.

## First-run diagnostic (2026-06-13): the trust region is `lr`, not `kl_coef`

The first probe shard (K=128, lr=0.05, steps=3, `kl_coef=1.0`) read **−247.60/round**
vs the non-adapted θ_o control (n=100, 93% loss, call-bonus −96) — not flat, a
catastrophic regression. A `diagnose` pass (`scripts/diag_pmcpa.py`) measured, across
real round-start adaptations, the global drift `KL(θ_a‖θ_o)` on held-out decision
states and the fraction of held-out argmax actions that flip:

| kl_coef | lr | steps | KL_glob | flip% |
|---|---|---|---|---|
| 1 | 0.05 | 3 | 1.909 | 51.7 |
| 300 | 0.05 | 3 | 1.476 | 49.1 |
| 1 | 0.01 | 3 | **0.043** | **7.4** |
| 1 | 0.05 | 1 | 0.911 | 32.4 |

**Findings:**
- **`kl_coef` is nearly impotent** — 1 → 300 barely moves drift (1.91 → 1.48). The
  update uses **Adam**, whose per-parameter step normalization renormalizes a larger
  soft KL penalty away; the anchor coefficient is *not* the trust region.
- **`lr` is the trust region** — 0.05 → 0.01 collapses global drift 44× (1.909 →
  0.043) and argmax-flips 52% → 7%. At lr=0.05 the head saturates toward the biased
  K-world targets and corrupts play on **unrelated** states (half of all decisions
  flip), because `kl_anchor_loss` constrains only the *sampled* states.
- The anchor is also too **local**: `KL_loc` (5.8) ≫ `KL_glob` (1.9).

**Fix applied:** default `lr` **0.05 → 0.01** (`adapt_play_net`, the launcher, the
`.ps1`); `kl_coef` left as a weak secondary knob with a note that it is not the trust
region. A re-run at lr=0.01 (K=128, n=20) read **−94.50 [−196, +3.5]** (call-bonus
−40): the catastrophe is removed and drift sits in vine's EV-neutral band, but the
point estimate is still negative and n=20 is too noisy to separate "≈ flat" from
"mildly negative (danger-(b))". **Throughput wall:**
~22 min/Position measured → the faithful K=128 probe is ~6 days.

**Powered read + oracle decider (2026-06-13).** A powered gate at the fixed config
(lr=0.01, K=128) read **−59.33 [−114.68, −6.16]** at n=60 — CI excludes 0, a confirmed
regression, not flat. To decide whether this was world-sampling bias (danger-(b),
fixable by belief-on / mid-round re-adapt / more worlds) or a broken mechanism, an
**oracle** variant (`PMCPAOracleAgent`, `worlds_override`, the runner's
`on_round_start_oracle` hook) adapted on the **true GameState** — danger-(b) = 0, the
ceiling of any world-quality fix, and ~45× cheaper (K=1). Result (n=200):
**−67.83 [−94.24, −39.25]**, no better than the sampled −59. **Even a perfect per-deal
gradient degrades play**, so world quality is not the gap and every (b)-rescue is ruled
out. The mechanism is structurally unsound for the reason in the closure banner: the
within-deal paired advantage assumes a θ_o continuation the adapted θ_a abandons. This
is the decisive, faithful-mechanism kill (the oracle is an upper bound — a negative
there cannot be a proxy false-negative). The `lr` fix and the `kl_coef`-is-not-the-
trust-region finding stand as the secondary diagnostic record.

## Reuse map

| Component | Source | Verdict |
|---|---|---|
| Paired-advantage rows / playout / candidates | `ppo/vine.py`, `search/blunder_miner.py` | reuse; root at a real round instead of dedicated vine games |
| Determinized-World sampler (belief-off/on) | `search/determinize.py` `sample_determinized_world` | reuse as-is |
| In-memory net → Agent | `MLAgent.from_loaded` | reuse as-is |
| Full-round driver + round-start hook | `tichu_eval/play_full.play_full_round` | reuse; **add** optional `on_round_start` |
| A/B tournament + seat-swap + CI | `cli/eval_matrix`, vine A/B configs | reuse as-is |
| `PMCPAAgent` (per-round clone + optimizer + adapt) | — | **new** |
