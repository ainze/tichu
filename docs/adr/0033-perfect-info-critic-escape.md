---
status: proposed
---

# Reopen the strength program via a Perfect-Info Critic (PTIE), gated by a cheap value-ceiling test

[ADR-0031](0031-search-and-learning-loop.md) and [ADR-0032](0032-owner-free-ev-probe-claim-solver.md)
closed the strength program: every self-play / search / RL lever was falsified, and
five convergent instruments (four EV-probes + the decile-9 move-prediction audit)
showed the BC `master` plays the measured decisions correctly. The standing
conclusion was **accept BC(+PIMC) and re-scope away from "superhuman"** (referred to
in those ADRs as "option 4").

This ADR records a **deliberate reopening** of that conclusion, decided in the
grilling + deep-research session of 2026-06-07. It does **not** lock the build; it
records *why* the closed question is reopened, the *one new instrument* that justifies
it, the cheap **gate** that must pass first, and the falsification criteria. The
vocabulary (**Perfect-Info Critic**, **Perfect-Info Value-Ceiling Test**) is glossed in
[CONTEXT.md](../../CONTEXT.md) §"Phase 2 / online-RL terms".

## Terminology note — two colliding "option 4"s

The handoff that opened this session and the prior ADRs use "option 4" for **opposite**
things. This ADR retires the overloaded label and uses:

- **ESCAPE** — pursue a fundamentally different, *non-imitation* method that can exceed
  the decile-9 ceiling (the handoff's "option 4").
- **ACCEPT** — ship BC(+PIMC), re-scope away from "superhuman", possibly deploy the
  **Claim Solver** as an endgame safety-net (ADR-0031/0032's "option 4").

## Why reopen — the convergence evidence shares one ceiling

The five instruments do **not** establish "the master is strong." Every one is either
**self-relative** (the four EV-probes measure *vs master*) or **imitation-capped at
decile-9** (the move-prediction audit measures imitation of decile-9 humans). If a
genuinely superhuman line exists, **none of them can see it** — this is ADR-0032's own
Risk 4 ("the decile-9 / owner blind spot"). So the convergence supports only the
narrower claim: *the master imitates decile-9 well and the specific forced behaviors
are not better than master.* It does not foreclose a non-imitation method that exceeds
decile-9.

## The one new instrument that justifies reopening

A deep-research pass (single-machine + 1–4 GPU envelope; 24/25 claims verified 3-0)
surfaced **PerfectDou's Perfect-Training-Imperfect-Execution (PTIE)** as the
highest-evidence-to-compute lever for a Doudizhu-like game (the closest structural
analog to Tichu), and — critically — one this project has **not** tried:

1. **A Perfect-Info Critic is strategy-fusion-immune by construction.** The value net
   sees all four hands at training time; the **Policy Network** sees only the info-set
   and is the only thing at inference. The deployed policy cannot select per-world
   divergent actions — the exact defect that broke **PIMC** and the **Search + Learning
   loop**. This is a *different* immunity mechanism than ReBeL/Student-of-Games (which
   buy soundness via public-belief-states / GT-CFR but whose guarantees are proven only
   for two-player zero-sum — high transfer risk to 4-player partnership Tichu).
2. **Our flat PPO Refine v1 result has a now-identified confound.** [ADR-0029](0029-ppo-refine-design.md)
   used a **symmetric** critic — an MLP over the *same 224 observable features as the
   policy* (confirmed in `ppo/policy.py` / `ppo/update.py`: the critic is called as
   `critic(features)`). Its run note concluded *"value_loss floor is irreducible
   `round_outcome` variance, not critic ignorance"* — but that variance is irreducible
   **only given observable info**. A high-variance advantage is the textbook cause of a
   policy gradient going flat. We diagnosed "critic isn't the bottleneck" without ever
   testing the critic variant that addresses it. So PPO-flat is **weak evidence**
   against PTIE specifically.
3. **It learns from self-play, not imitation** → not decile-9-capped → the only class
   of in-scope method that can, in principle, exceed top human play.
4. **The single-box objection to from-scratch RL was factually wrong.** ADR-0029
   rejected DouZero-DMC as "cluster scale"; the research shows **DouZero+ trained on
   one RTX 2080Ti**, and PerfectDou is ~10× more sample-efficient than DouZero.

## Measurement — we already own the validated instrument

Exploitability / NashConv is confirmed **intractable** at this scale (DouZero's authors
say so). Every cited project — DouZero, DouZero+, PerfectDou, bridge's alpha-mu —
verified gains owner-free via **duplicate / seat-swap dealing scored by WP + ADP vs
prior-SOTA bots**. That is exactly the existing **Tournament + Seat-Swap Matrix**. A
self-play agent that **decisively beats `master` in seat-swapped tournament** is a real
break past the imitation ceiling (it did not learn by imitating). It cannot certify
"superhuman vs the owner" in absolute terms, but "exceeds the decile-9 ceiling" is the
measurable, fundable milestone; the owner play-test remains the cheap out-of-distribution
backstop, as in ADR-0029.

## Decision

**Reopen ESCAPE via a Perfect-Info Critic driving PPO self-play — but gate the
multi-week build behind a cheap, decisive `Perfect-Info Value-Ceiling Test` first.**
Do not build the PPO loop until the gate passes. This mirrors the project's proven MO
(the value-ceiling sweep before loop compute; the EV-probes before each escalation):
falsify the premise cheaply before paying for the build.

### The gate — `Perfect-Info Value-Ceiling Test`

- **Data:** master self-play, distribution-matched to the eventual PTIE critic
  (extend `scripts/value_ceiling_sweep.py`, whose `observer` already captures
  `(features, z)`; add the ground-truth opponent hands from the engine `GameState`).
  Generate enough rounds that the *observable* R² has **plateaued** w.r.t. data (the
  prior self-play sweep was starved at ~85k) before trusting the delta.
- **Two critics, same data, round-level split:** observable-224 vs perfect-info-392
  (the 224 **Feature Vector** + opponents' true hands on the **Belief Model**'s
  `(3 × 56)` relative-seat grid).
- **Metric:** residual-variance ratio **ρ = (1 − R²_perfect) / (1 − R²_observable)** —
  the share of policy-gradient advantage variance that survives once the critic sees
  all hands. Reported **three ways**: overall, on **call-live states** (a seat that has
  called Tichu/Grand — highest-variance, partnership-relevant, the build's #1 untested
  risk), and the rest.
- **Pre-committed bands:**
  - **ρ ≤ 0.5 → BUILD.** Perfect info at least halves advantage variance; the
    symmetric-critic confound is real and removed; the PTIE-PPO build is justified.
  - **ρ ≥ 0.8 → KILL ESCAPE → ACCEPT.** Perfect info buys ≤20%; the symmetric critic
    was not the confound; ship BC(+PIMC), re-scope, consider deploying the Claim Solver.
  - **0.5 < ρ < 0.8 → AMBER.** Run a **short asymmetric-critic PPO smoke** (does
    advantage variance drop *and* does the policy commit to bombs/presses without
    KL-collapse?); the call-state breakout breaks the tie.

ρ ≤ 0.5 is **necessary, not sufficient** — it proves the diagnosed confound is real and
removed, not that PPO reaches **Master-level (goal)**. That the build itself discovers.

### Build (only if the gate passes) — design intent, not yet locked

- **Perfect-Info Critic** (PTIE) + PPO self-play, reusing the ADR-0029 harness
  (vectorized rollout, GAE, KL-anchor, league); the genuinely new pieces are the
  train-time perfect-info featurizer and the asymmetric critic.
- **Widen ADR-0029's self-admittedly-too-narrow scope:** include the **call sub-game**
  (the highest-variance decision, frozen in v1) rather than play-head-only; reconsider
  on-turn-only.
- **Measure** with the existing seat-swap **Tournament Matrix** vs the full agent zoo;
  owner play-test as the OOD backstop.

## Rejected alternatives

- **ACCEPT now (the ADR-0031/0032 conclusion).** Premature: the convergence evidence is
  decile-9-capped/self-relative, and PTIE is the first genuinely-untried, non-imitation,
  fusion-immune, single-box lever — with our flat-PPO counter-evidence now confounded.
  ACCEPT remains the destination *iff* the gate comes back RED.
- **DouZero-style DMC from scratch.** Now known single-box-feasible and the most-proven
  precedent on the closest game, but discards the BC investment and reuses less of the
  existing stack; PTIE-PPO has the better evidence-to-compute ratio for *this* codebase.
  Held as the fallback if warm-started PTIE-PPO stalls.
- **ReBeL (public belief states) / Student of Games (GT-CFR).** Game-theoretically sound
  and superhuman in poker, but guarantees are proven only for two-player zero-sum
  (ReBeL) or validated on 2–3-player non-partnership games (SoG, whose convergence proof
  was explicitly refuted in verification). High transfer risk to 4-player partnership
  Tichu with a call sub-game; not justified before the cheaper PTIE lever is tested.
- **alpha-mu (depth-extended PIMC).** Provably corrects fusion but only under
  perfect-information-for-opponents (the same assumption family that hurt our PIMC line),
  with empirically small gains. Not compelling.

## Consequences

- Re-enters the online-RL regime that ADR-0031 "Final synthesis" closed. The surprise of
  that reversal — against five convergent instruments — is the reason this ADR exists;
  the justification is one new instrument (PTIE) + one identified confound (the symmetric
  critic), not a re-litigation of the refuted passivity diagnosis.
- The first deliverable is the **gate**, not the loop — its cost is dominated by cheap
  master self-play and it can route straight to ACCEPT if RED.
- Do **not** resume the refuted lines: the caller-passivity / bomb-undervaluation
  diagnosis, PIMC-as-policy-target, or the search+learning loop.

## Risks

1. **Partnership + call sub-game is untested in all cited work.** Doudizhu has no fixed
   partnership across a hidden-info boundary, no schupfen, no binary call sub-game. PTIE's
   fusion-immunity is structural and transfers, but its *effectiveness* on a partnership +
   call game is the genuine unknown — partially previewed by the gate's call-state ρ, fully
   resolved only by the build. (The research's #1 open question.)
- 2. **R² is a necessary-not-sufficient proxy.** A clean advantage signal does not
   guarantee PPO improvement; the AMBER smoke exists to catch a high-ρ-but-still-flat case
   before the full build.
3. **Self-play distribution drift.** The gate measures on master self-play; the eventual
   PTIE policy drifts off-distribution, where the critic must keep up via online fitting —
   standard, but a watched signal.
4. **The owner blind spot persists.** A built agent that beats master in tournament
   exceeds the *imitation* ceiling but still cannot be *certified* superhuman owner-free;
   the owner play-test is a vibe-check, not a measurement.

## Success criteria

The gate produces a clear verdict: **ρ ≤ 0.5** justifies the build (a real, located reason
the flat-PPO result was confounded), or **ρ ≥ 0.8** falsifies the PTIE premise cheaply and
routes to ACCEPT — the first principled basis for accepting the ceiling that does not
itself share the decile-9 ceiling. The eventual build ships a **Sharpened Checkpoint** only
if it beats `master` on the seat-swap **Tournament Matrix** with CI excluding 0.

## Result — the offline-R² gate is structurally confounded; pivot to a PPO smoke (2026-06-07)

The gate was built test-first (`src/tichu_training/perfect_info.py` —
`featurize_perfect_info`, `is_call_live`, `r2_score`, `residual_variance_ratio`,
`split_rho`; an additive `state_observer` hook on `play_full_round`; the orchestration
script `scripts/perfect_info_value_ceiling.py`; an `n=20000` position pool) and run on
**1.13M master self-play samples**. It did **not** yield a trustworthy verdict — running
it exposed two structural confounds that make the offline held-out-R² proxy the **wrong
instrument** for an online privileged critic:

| | observable | perfect-info |
|---|---|---|
| train R² | 0.701 | **0.848** |
| test R² (held-out rounds) | 0.190 | 0.123 |

1. **No bug; the information is real.** Perfect-info **train** R² (0.848) exceeds observable
   (0.701) — the superset relationship that must hold, confirming the features are correct.
2. **Determinism (over-states perfect-info).** `master` plays **greedy** (`_act_on_play`
   returns `ranked[0]`), so `round_outcome` is a *deterministic function of the full
   GameState* → perfect-info's ceiling is ~1.0 only because there is no policy stochasticity
   to resolve. The eventual PTIE critic faces a *stochastic, changing* PPO policy, so this
   inflates the metric.
3. **Round-unique privileged features (under-states perfect-info).** Every Round is a unique
   4-hand deal, so the 168 opponent-hand dims are *round-constant*; under a round-level
   held-out split they are intrinsically memorization-prone (a hand configuration essentially
   never recurs). Perfect-info therefore **overfits harder** (train–test gap 0.73 vs 0.51)
   and its *test* R² collapses below observable. But an **online PPO critic is used
   on-distribution** (refit continuously on fresh rollouts, used as a baseline on that same
   data) — it never needs to generalize to unseen rounds, so held-out failure does **not**
   imply online failure. The relevant on-distribution figure is ρ_train ≈ (1−0.848)/(1−0.701)
   = **0.51** (contaminated, but mildly favorable), not the held-out ρ ≈ 1.08 the script printed.

The instrument cannot pin the real number (train-ρ optimistic, test-ρ pessimistic, determinism
muddies both), so **neither the printed "KILL" nor a "BUILD" is trustworthy**. The honest
conclusion is that the cheap offline-R² gate is mismatched to the online-critic question, and
patching it (stochastic self-play + early-stopping) would not fix the held-out-memorization
mismatch.

**Decision (grilling 2026-06-07): pivot to the direct test — an asymmetric-critic PPO smoke**
(the ADR's own AMBER path, promoted to the decider). Reuse the ADR-0029 PPO harness unchanged
except the critic sees **perfect-info** features (392) while the policy stays observable — a
clean A/B against the flat symmetric run (+2.1, CI [−4.74, +9.08]). It measures the actual
claim (does a perfect-info critic reduce advantage variance and let PPO move) without the
offline-fit confounds, and it is most of the BUILD, so it does double duty.

- **Build:** separate policy-features (224, for the policy re-forward) from critic-features
  (392, for the value) through rollout → trajectory → batch → update (additive
  `critic_features`, default None → symmetric fallback; thread the `GameState` into
  `act_play_batch`). Perfect-info critic warm-started on the cached 1.13M self-play samples
  (free strong leaf from iter 0 — the PerfectDou-faithful setup).
- **Decision criteria:** PRIMARY = Tournament-vs-master EV (BUILD if CI excludes 0 positive
  / clearly beats the flat run; KILL-lean if flat within noise *despite* a strong perfect
  critic — echoing ADR-0031's "a good value alone did not move EV", but now without the
  strategy-fusion target). MECHANISM = online value R² markedly above the symmetric critic.
  BEHAVIOR = `caller_bomb_passivity` (flat plateaued ~0.285) moves, `bomb_when_legal` bounded.
- **Reusable artifacts kept:** `perfect_info.py` (`featurize_perfect_info` is exactly the
  critic's train-time input), the `state_observer` hook, the `n=20000` pool.

### Build — asymmetric critic wired through the PPO harness, test-first (2026-06-07)

The Perfect-Info Critic is built into the ADR-0029 PPO harness as an **additive** option
(`critic_features` defaults to `None` → symmetric fallback → the flat run is byte-identical):

- **Data flow:** policy-features (224, observable) are separated from critic-features (392,
  perfect-info) through `rollout → trajectory → batch → update`. The rollout threads the
  `GameState` into `act_play_batch` (`(seat, private_state, game_state)`); `BatchedPolicy(
  perfect_info=True)` builds `featurize_perfect_info(game_state, seat)` for the critic;
  `PPOBatch.critic_features` carries them; `ppo_update` values `critic(critic_features)` while
  the policy re-forwards on observable `features`. `train_ppo(perfect_info=...)`, `League(
  perfect_info=...)`, and the `_critic_warmup` all thread it.
- **Tests:** 7 new behaviors in `tests/training/ppo/test_perfect_info_critic.py` (field
  plumbing, GameState passing, 392-value, symmetric regression, batch separation, update
  shape, end-to-end). Full PPO suite 40 green; touched-area regression green. (10 pre-existing
  full-suite failures are unrelated test-ordering pollution: `forced_*`/`search`/`ml` contract
  tests need a `checkpoint_path` — flagged separately.)
- **Run harness:** `configs/ppo_v5_perfect_info.yaml` (the A/B: identical to `ppo_v5.yaml`
  except `perfect_info: true` + a 20-iter critic warm-up) and `..._smoke.yaml`. The smoke ran
  end-to-end on the real `master` — perfect-info warm-up, asymmetric PPO update, a league of
  perfect-info opponents (no shape clash), snapshots, and a round-tripping Checkpoint.

The A/B run is the decider per the criteria above. The exported policy is observable-only (the
perfect-info critic is train-time-only, never shipped), so the existing seat-swap **Tournament
Matrix** evaluates it unchanged.

### A/B result — the Perfect-Info Critic does NOT unstick PPO → ACCEPT (2026-06-07)

The run was externally killed at iter 163 (no traceback — OOM / sleep, not a code fault; 20
warm-up + 163 PPO iters ran clean), leaving a complete `iter_00160` snapshot — well past the
flat run's ~iter-70 plateau, and the loss curves showed this policy had likewise plateaued
(KL pinned at the 0.04 target, `policy_loss` ≈ −0.002, entropy flat from ~iter 30). The
mechanism signal was already weak: `value_loss` ~5.1k vs the flat run's ~6–7k floor — only a
modest cut, because in *stochastic* self-play the outcome variance is dominated by the sampled
future (which perfect info cannot resolve), not the hidden hands. So iter_160 was exported and
tournamented (n=4000 seat-swapped, `configs/eval_tournament_ppo_perfect_info.yaml`):

| pair | mean delta | read |
|---|---|---|
| **master vs pi_iter160** | **−3.7** (call-bonus −1.2) | perfect-info **flat vs master**, within noise |
| master vs ppo_flat_iter150 | −1.6 | the symmetric flat run, same |
| pi_iter160 vs ppo_flat_iter150 | +1.7 | the two are **indistinguishable** |

**Verdict: KILL ESCAPE → ACCEPT.** The asymmetric perfect-info critic — the hypothesized
confound behind flat PPO — is *not* the confound; a critic that sees all four hands does not
make PPO win. This is the cleanest refutation yet of "the value/critic is the bottleneck"
(now in PPO, with no strategy-fusion target and a perfect critic). It is the ~6th flat
self-play/RL result.

**Status: this ADR's ESCAPE hypothesis is refuted.** The honest end is **ACCEPT** — ship the
BC `master` (+ the deployable **Claim Solver** endgame net), re-scope away from "superhuman".
One structural lever remains genuinely untested and is the only thing that could reopen ESCAPE:
**joint co-training of play with the frozen subsystems** (Schupfen / Call Networks) — the
hypothesis that a stronger play line is unreachable while it best-responds to *master's*
schupfen-shaped hands and call policy (a frozen-subsystem local optimum). Cheaply probe-able
*before* any co-training build via the `forced_*` EV-probe harness (a heuristic-schupfen probe;
a Claim-Solver-driven extra-Tichu-call probe). If both land flat, the interaction hypothesis is
falsified and ACCEPT is unconditional. (The **Belief Model** is *not* part of this — it is a
predictor, not a policy, and the perfect-info critic already dominates it with ground-truth hands.)

### Coupling probes — calls flat, schupfen sensitive-but-unsettled (2026-06-07)

Both probes read as **sensitivity** tests, not +EV hunts (forcing a fixed-play agent to
schupfen/call differently breaks the BC co-adaptation, so deltas are expected ≤ 0; the signal
is the *magnitude*). The Claim-Solver call idea was dropped — calls fire at round start with
full hands, where no Guaranteed Out is detectable — and replaced by the cleaner `tichu_threshold`
knob (a calibration knob, so +EV *is* a legitimate outcome there).

- **Call coupling (`configs/eval_tournament_call_coupling.yaml`, n=4000).** Tichu-call rate swept
  threshold 0.20 → 0.70, master play held fixed; **every delta vs master within ±1.2, CI spans 0**
  (call_more 0.35 → +0.43; call_max 0.20 → +0.89; call_less 0.70 → +1.20; extremes 0.20-vs-0.70 → −0.4).
  The EV-vs-call-rate curve is **flat** → the call head is not miscalibrated *and* call rate does
  not move play EV → **calls are not a coupling lever.**
- **Schupfen coupling (`configs/eval_tournament_schupfen_coupling.yaml`, n=4000;
  `ForcedSchupfenAgent`).** master play + alternative schupfen: **`schupfen_rule` −51.8** (CI
  [−58.4, −44.7]), **`schupfen_random` −142.9** (floor). Play EV is **highly sensitive** to
  schupfen, and **most of the −52 flows through the call bonus** (−38) — concrete confirmation
  that schupfen↔call↔outcome are coupled. But sensitivity is *necessary-not-sufficient*: it shows
  master's *learned* schupfen is far better than heuristics (BC cloned play+schupfen *jointly* from
  humans → already co-adapted at decile-9), **not** that a *better-than-BC* joint optimum exists.
  No cheap test can settle that — only co-training can.

**Decision (2026-06-07): pursue full-stack co-training.** The cheap probes falsified the call
lever and (with the perfect-info A/B) the critic lever, but could not falsify the schupfen-coupling
hypothesis — it is the *one* non-flat signal. The owner elected to settle it by building the thing
the whole project deferred: **a single PPO that co-trains play + Schupfen + Tichu/Grand calls
together** (the Belief Model excluded — it is a predictor). This is the project's hardest build
(schupfen is a one-shot round-start decision with the longest credit horizon; calls add ±100/±200
variance; both are *separate* networks from the shared-trunk play head) and runs against ~6 flat
RL results, so it must be designed (grilled) before building, with a pre-committed kill budget and
the same ship bar: beat `master` on the seat-swap **Tournament Matrix** with CI excluding 0.
Handoff for the next session: `%TEMP%\handoff-tichu-fullstack-ppo-cotrain.md`.
