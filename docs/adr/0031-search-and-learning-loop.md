---
status: proposed
---

# Phase-2 search + learning: AlphaZero-style policy+value co-training on search targets

[ADR-0030](0030-phase-2-search-design.md) built the PIMC search stack and ran it on
frozen nets. The decisive result, then firmed at n=120: **search-only on a frozen
master-derived value does not fix caller passivity — at depth it amplifies it**
(deep critic-search 0.323 / 0.263 vs master 0.162). The exploration-critic pilot
(0030, "exploration-critic pilot") tested the cheapest escape — refit only the
*value* on an aggression-exploring distribution — and it **failed to clear master**
(0.263 vs 0.162, ~3.5σ worse). That isolates the conclusion: the residual passivity
lives in the **policy/prior** (which seeds PUCT and supplies the in-tree environment
moves), not only the value, so no value-only intervention can reach it.

The only mechanism left that structurally matches the goal is the one 0030 reserved:
**search as a target generator for learning** — discover value *and* policy by search,
retrain both on search-improved targets, iterate. This ADR scopes that loop. It does
**not** lock numbers; it records the design, the reuse map, the phasing, and the
risks so the build (and a grilling pass) can proceed from a shared picture.

## The loop (one generation)

Given frozen artifacts `(policy_g, critic_g)`:

1. **Self-play with search.** Play N rounds from the position pool. Every seat acts by
   PIMC over `(policy_g` as prior + in-tree env, `critic_g` as leaf)`, with in-loop
   exploration (below). At each Play Decision record `(features, legal_mask, π)` where
   **π = normalised root visit counts** over legal play actions — the policy-improvement
   target. At round end attach the **team-relative `round_outcome` (/100, call bonus
   included)** as the value target `z` to every recorded decision of that team.
2. **Train.** From the buffer:
   - **Play head** ← KL(π_target ‖ softmax(play_logits over legal)), warm-started from
     `policy_g`. *(New loss primitive — see Decision C.)*
   - **Value** ← MSE(critic(features), z), warm-started from `critic_g`, via the existing
     `fit_value_baseline`.
3. **Export** `policy_{g+1}.pt` + `critic_{g+1}.bin` through `export_torchscript`
   (version-stamped) so `MLAgent` / `SearchAgent` consume them unchanged.
4. **Evaluate** (two tiers — Decision J): *every generation*, the cheap raw-policy
   behavioral dial (`caller_passivity_rate` vs master 0.162) + the wish/dragon
   move-prediction guardrail; *at candidate nets*, the expensive `SearchAgent` strength
   tournament vs master (the −49 baseline) + search-behavioral — the real deployed product.
5. **Gate.** Promote `policy_{g+1}` to the next generation only if its **strength vs master
   does not regress** below `policy_g` (within tournament noise). Passivity is the headline
   *readout*, not the gate — it explains the strength curve but a less-passive-yet-weaker net
   is not shipped. If K≈3 consecutive generations fail to improve strength, **stop and
   diagnose** rather than spin. Track both curves across generations.

The terminal `round_outcome` is the ground-truth signal that breaks the passivity cycle:
it already rewards pressing a lead that wins and penalises a failed call. As long as
exploration surfaces the aggressive states, their true outcomes pull the value up and the
visit targets (hence the policy) follow. The passivity dial per generation is the monitor
that this is actually happening and not collapsing back to the passive prior.

## Decisions

Decisions A–K were walked one-by-one in a grilling pass (2026-06-04). The core design was
unchanged; the pass concretized every open branch and recorded the deferred (Phase-2) knobs.

**A. Self-play engine — base the collector on `play_full_round`, not `ppo/rollout.py`.**
*(Revised at Phase-1 implementation, 2026-06-04.)* The original plan reused
`collect_rollout` + `RolloutPolicy`, but that harness **deliberately drops the Tichu Call**
("Calls are out of PPO scope") and sets only Grand callers. That is disqualifying here: the
pathology is *caller* passivity, and the ±100/±200 call bonus in `round_outcome` is the
exact signal that punishes a failed call — so the targets must come from rounds where Tichu
calls actually fire. `tichu_eval.play_full.play_full_round` drives the full
call/schupfen/play/wish/dragon stack with the call bonus folded into the outcome, and
`fit_search_critic` already uses its `observer → features` + `result.total → z` pattern. So
the Phase-1 collector runs `play_full_round` with four recording search agents; each agent
computes the **visit distribution π** in `act` and records `(features, π)` per Play
Decision, and the collector attaches the team-relative `z` at round end. `ppo/rollout.py`'s
batched `act_play_batch` protocol is revisited for the S2 throughput engine in Phase 2.

**B. Throughput is the existential risk — phase it (S1 → S2).** PIMC self-play is the
cost wall: the deep behavioral run was ≈4 min/round at 360 sims × 4 seats. Thousands of
rounds serially is infeasible.
- **S1 (Phase 1): decision-level, no inner batching.** Each decision runs a full
  single-tree MCTS (`mcts.run_mcts` as-is) at the reduced Phase-1 budget (Decision K), with
  **multiprocessing** (the `eval_matrix` Pool pattern). Goal: prove the *learning signal*
  at small scale before optimising.
- **S2 (Phase 2, only if S1 signal is positive): batched-MCTS.** Run M trees in lockstep;
  at each sim step gather the leaves-to-expand across all M trees and issue **one** batched
  policy+critic forward. This is the real AlphaZero throughput pattern and the
  vectorisation the ADR-0029 harness reserved ("vectorized many-game stepping is a later
  slice that preserves this interface"). It needs a new batched-MCTS engine; `run_mcts` is
  single-tree sequential today.

**C. Policy target = raw aggregated visit-proportions, trained with a new KL loss.**
π(a) = `agg_visits(a) / Σ agg_visits` over the **legal** play actions (summed across the K
worlds), **no temperature sharpening** (sharpening amplifies the noise of a few visits into
a near-one-hot wrong target), and **forced single-legal-action decisions are skipped** (no
policy signal). `bc/loss.masked_cross_entropy` is hard-label only (`target` is a scalar
index), so add `masked_kl_divergence(logits, target_probs, legal_mask, sample_weight)`
taking a `(B,K)` distribution — the **only new training primitive**. Phase-1 low-sim π is
accepted as noisy and tamed by dataset averaging, not over-engineered; if too sparse, the
reserved levers are more sims (Decision K) or mixing a small prior floor into π.

**D. Value target = Monte-Carlo `round_outcome`, no bootstrap.** Scalar team-relative
outcome (/100, call bonus included), the same `z` attached to every recorded state of that
team in the round (correct for a state-value regressor), fit by `fit_value_baseline` as-is
(MSE/Adam). MC is AlphaZero-faithful and is the *true terminal signal* that breaks the
passive value — bootstrapping on the current critic would re-inject the bias we fight.
Lower-variance variants (root-value bootstrap, λ-returns) are a Phase-2 knob only if value
training proves too noisy.

**E. Single-rooted target generator; all four seats learn.** Every seat searches with the
shared `policy_g` as prior+env and `critic_g` as leaf, so every Play Decision yields a
target (4× data). Inside the tree the other three seats (incl. partner) are advanced by the
policy, not by their own search — the same approximation `SearchAgent` v1 makes. The policy
thus learns a best-response to a non-searching opponent model; the bias is bounded and is
exactly what the per-generation strength gate (Decision J) is there to catch. Modelling
in-tree opponents as *also-searching* (nested search) was rejected for v1 — it trades the
whole feasibility story for fidelity not yet shown to be needed.

**F. In-tree world fidelity: belief OFF, opponents SAMPLED (τ=1).** Determinization stays
**belief-off (uniform worlds)** for Phase 1 — matches every measurement so far, adds no
second moving part; **belief-on is the first Phase-2 fidelity lever** if targets are noisy.
But the loop's `EngineWorld` advances in-tree opponents by **sampling** `play_action_scores`
(τ=1), *not* greedy argmax — greedy best-responds to a single deterministic line (a brittle,
exploitable training target), sampling makes the visit target "good against the opponent's
distribution." Cost is identical (same forward). The frozen-experiment `EngineWorld` keeps
greedy via a flag for reproducibility.

**G. In-loop exploration — Dirichlet root noise + visit-temperature, concrete values.**
At *every* decision's root, `prior ← (1−ε)·prior + ε·Dir(α)` with **α=1.0, ε=0.25** — the
load-bearing knob that floors the suppressed bombs/beats so they get visited and valued by
their true outcome. Advance self-play by sampling `∝ N(a)^(1/τ)` with **τ=1 throughout the
round** in Phase 1 (short rounds; coverage over strong play, since strength is measured
separately on the greedy-deployed net). τ-annealing and α-tuning are Phase-2 refinements.
The pilot's `ExplorationAgent` is subsumed by this.

**H. Train the play head + value; the shared-trunk update is (ii) trunk+play with a
guardrail.** `BCModel` is a shared trunk → {play, wish, dragon, …} heads; the value net and
the schupfen/call nets are separate modules (untouched). Training the play head on π
backprops through the trunk, so wish/dragon are at risk. We **train trunk + play head**
(option ii — passivity plausibly lives in the representation, so head-only is too weak) and
**guard** by re-running the move-prediction eval on wish/dragon each generation; if they
degrade materially, escalate to a multi-task BC-anchor (option iii). Schupfen/Tichu/Grand
stay frozen — the pathology is *play after calling*, squarely the play head. Co-adapting the
call heads is a later, unscoped pass.

**I. Warm-start every generation; current-generation replay only.** Each generation
warm-starts from `(policy_g, critic_g)` with few epochs / small LR (a gentle nudge that
preserves BC competence and stays stable on thin data). Train on the **current generation's
games only** — clean on-policy signal, no staleness confound when reading whether passivity
dropped *because of* the loop. A sliding window (W≈2–3 generations) is the Phase-2
data-efficiency lever once full-scale generation is the cost.

**J. Gate on strength-non-regression; passivity is the headline; two-tier eval.** Promote
`policy_{g+1}` only if EV-vs-master ≥ `policy_g` (master is the actual goal and is
comparable across generations); a less-passive-but-weaker net is not shipped. Passivity (vs
0.162) is the headline readout that explains the strength curve. Eval is two-tier: the cheap
raw-policy dial + wish/dragon guardrail *every* generation, the expensive `SearchAgent`
strength tournament + search-behavioral *at candidates* (the real deployed product). 3-gen
stall fallback (Decision in loop step 5).

**K. Phase-1 scale — a direction-finder, not a strength run.** 4 worlds × 50 sims
(200/decision), ~50 rounds/generation (≈7k play-decision samples after skipping forced
moves), **3 generations**, sharded across 6–8 spawn processes (torch threads pinned to 1),
**co-training policy and value** each generation. Estimated ~1.5 h for 3 gens. Framing: if 3
gens show no downward passivity movement, the *first* response is more rounds + more sims
(cheap), and only if that also fails do we conclude the loop is wrong — this ordering stops
an under-powered run from false-negativing a sound approach.

## Reuse map

| Component | Source | Verdict |
|---|---|---|
| Determinize / MCTS / EngineWorld / Critic load+save | `tichu_training/search/*` | reuse as-is |
| Self-play round driver + per-decision recording | `ppo/rollout.py` `collect_rollout` + `RolloutPolicy` | reuse skeleton; add visit-π field + `SearchRolloutPolicy` |
| Policy arch + training loop | `bc/training.py`, `BCModel` | reuse; **add** `masked_kl_divergence` |
| Value training | `awr/value_baseline.fit_value_baseline` | reuse as-is |
| Export → TorchScript (version-stamped) | `tichu_export.export_torchscript`, `cli/export_model` | reuse as-is |
| Per-generation eval + gating | `cli/eval_matrix` behavioral + tournament | reuse as-is |
| Batched-MCTS engine (S2) | — | **new** (Phase 2) |

## Phasing

- **Phase 0 — primitives (~0.5d).** `masked_kl_divergence` + unit test; extend
  `PlayChoice`/trajectory to carry π; a sampling+Dirichlet `EngineWorld` variant (Decisions
  F, G); a `SearchRolloutPolicy` that emits `(features, π, z)`.
- **Phase 1 — one loop, small scale (~2–3d).** S1 budget at the Decision-K scale (4×50, ~50
  rounds/gen, **3 generations**, 6–8 procs), co-training policy+value, warm-started.
  **Decision gate:** does `caller_passivity_rate` move *down* across generations with
  strength non-regressing (Decision J)? If flat, first raise rounds/sims (Decision K), then
  diagnose — before investing in throughput.
- **Phase 2 — throughput + scale (~3–5d, only if Phase 1 positive).** Batched-MCTS (S2);
  then the reserved fidelity/efficiency levers as needed — belief-on (F), sliding-window
  replay (I), τ-annealing (G); full-budget generations to convergence.
- **Phase 3 — iterate** to master-level / passivity < 0.162.

## Phase 0 build (2026-06-04)

The four primitives landed test-first (10 new tests, all green; existing search/BC suites
unchanged — the new knobs default off, so the frozen path is byte-for-byte the same):

- **`masked_kl_divergence`** (`bc/loss.py`) — soft-target KL for the play head; verified
  zero-at-match, illegal-masked, one-hot-equals-`masked_cross_entropy`, weight-scaling.
- **`sample_from_scores` + `EngineWorld(opponent_sample=True)`** (`search/engine_world.py`)
  — τ=1 in-tree opponent sampling (Decision F); greedy remains the default.
- **`run_mcts(..., root_noise=(alpha, eps))`** (`search/mcts.py`) — root Dirichlet noise via
  `rng.gammavariate` (Decision G); `None` reproduces frozen search.
- **`visit_policy`** (`search/mcts.py`) — normalises aggregated root visits into the π
  target (Decision C).

## Phase 1 build (2026-06-04)

The generation loop is assembled and smoke-tested end-to-end on the real master (16 new
tests across `test_selfplay.py`/`test_loop.py`/`test_mcts.py`, all green; 49 in the search +
loss suites total):

- **`SelfPlaySearchAgent`** (`search/selfplay.py`) — PIMC with root Dirichlet noise +
  opponent-sampling world, records `(features, π)` per Play Decision, advances by τ-sampling
  the visit distribution; calls/pending delegate to the policy. `play_target_vectors` maps π
  → `(target_probs, legal_mask)` over the action space.
- **`pimc_decide(root_noise=...)`** threads the exploration knob through to every world.
- **`search/loop.py`** — `collect_round` (run `play_full_round`, attach team-relative z),
  `samples_from_records`, and `train_play_head` (warm-start KL toward π through the shared
  trunk, Decision H).
- **`scripts/run_search_learning.py`** — the generation driver: load `(policy_g, critic_g)` →
  self-play → retrain play head + value (`fit_value_baseline`) → export `policy_{g+1}.pt` +
  `critic_{g+1}.bin`. `policy_0`'s trainable checkpoint is the BC final `step_000001.bin`,
  **verified byte-equal** to the master export (Δ play-logits = 0.0); `critic_0` is the
  master-self-play critic. Smoke (1 gen / 2 rounds) round-trips and re-loads cleanly.

The multiprocessing fan-out (`--workers`, spawn Pool over round-robin position shards) and a
per-generation raw-policy passivity dial + CSV log were added for the verdict run.

## Phase 1 result — the vicious cycle materialised (2026-06-04)

Ran **10 generations × 120 self-play rounds** at the Decision-K budget (4 worlds × 50 sims),
6 workers, ~11.7 h. Caller passivity (raw policy, fixed 200-deal held-out eval, n≈660–930
opportunities/gen) vs the gen-0 master baseline:

| gen | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `caller_passivity_rate` | **0.182** | 0.250 | 0.300 | 0.303 | 0.360 | 0.346 | 0.379 | 0.382 | 0.385 | 0.445 | **0.478** |

**Passivity rose almost monotonically, 0.182 → 0.478 (~2.6×).** The loop *amplified* the
pathology — the **vicious cycle** reserved as Risk 3 is real and quantified. Diagnostics:

- **Training works; the targets are passive.** Play-head KL fell within every generation
  (gen1 0.39→0.12, later ~0.06→0.04) — the policy faithfully moved toward the visit
  distribution. The visit distribution itself encodes passivity.
- **The value never learned — `value_mse` ≈ 4.5–5.5 every generation, with Var(z) ≈ 4.8, i.e.
  R² ≈ 0.** A near-uninformative leaf value means PUCT is **prior-driven**: the visit
  distribution ≈ the (passive) prior, training reinforces it, the next generation's self-play
  is more passive, the value stays uninformative — self-reinforcing. The small *systematic*
  passivity bias the value does carry (it is regressed from passive self-play) is what gives
  the drift its direction.
- **Exploration was too diluted to counter it** — Dirichlet ε=0.25 spread over 1809 actions
  rarely surfaces the specific caller-press lines, and even when it did, the R²≈0 value could
  not credit the aggressive outcome. `bomb_when_legal_rate` wobbled (0.097→0.11–0.14) without
  reducing passivity — more bombs in some spots, still ceding tricks overall.

**Verdict.** The cheap single-rooted configuration (belief-off, MC value regressed from the
loop's own increasingly-passive self-play, opponents modelled as the passive policy) does not
escape the passivity ceiling — it dives deeper into it. This is the **third** escalation to
fail the same way (search-only 0.323/0.263; value-only exploration-critic 0.263; now
policy+value 0.478), and the common thread is now unmistakable and measured: **the value
cannot credit aggression. R² ≈ 0 is the smoking gun.** No amount of policy-side loop
engineering fixes a value that carries no signal.

### Indicated next step — diagnose the value ceiling before any more loop compute

The run reframes the question from "how do we tune the loop?" to "**can *any* value model
predict `round_outcome` from the v5 features well enough to drive search?**" That is a cheap,
decisive diagnostic and must come first:

1. **Value-ceiling sweep (cheapest, most decisive).** Fit value models of varying data scale
   / capacity / target (MC vs a TD/bootstrap target) on `round_outcome` from the v5 featurized
   mid-round state, measure held-out R². If R² stays ≈ 0, the search-value approach is
   *structurally* capped and no loop variant will work — pivot (richer features, a different
   value signal, or accept the rollout-leaf ceiling). If R² can be pushed up, the loop has a
   path and points at which lever.
2. **If — and only if — value is learnable:** re-run the loop with (a) a **TD/bootstrap value
   target** (MC is too noisy → R²≈0), (b) **belief-on** worlds, (c) **targeted exploration**
   that biases toward legal beats/bombs in caller-following states (uniform Dirichlet is too
   diluted), and (d) the **Decision-J strength gate** active (it would have rejected gen 1 and
   halted the divergence — a safety net, not a cure).

The Phase-1 machinery (SelfPlaySearchAgent, the collector, the driver, the parallel fan-out)
is sound and reusable; the bottleneck is upstream of all of it, in the value.

### Value-ceiling result — data starvation, NOT a structural ceiling (2026-06-04)

`scripts/value_ceiling_sweep.py` fit value models on 84,701 master-self-play
`(features, z)` samples under a **round-level** 80/20 split (Var(z)≈4.73):

| model | train R² | test R² |    | train data (h=512) | train R² | test R² |
|---|---|---|---|---|---|---|
| linear (ridge) | 0.065 | 0.051 |    | 25 % (17k) | 0.110 | **−0.019** |
| MLP h=128 | 0.195 | 0.132 |    | 50 % (34k) | 0.155 | 0.066 |
| MLP h=512 | 0.377 | **0.232** |    | 100 % (68k) | 0.372 | **0.234** |
| MLP h=2048 | 0.431 | 0.232 |    | | | |

**The R²≈0 in the loop was data starvation, not an irreducible ceiling — this reverses the
tentative verdict above.** The data sweep is decisive: at **17k samples** (≈ the loop's
~10k/generation) test R² ≈ **0**, exactly what the loop saw; at **68k** it reaches **0.234**
and is *still rising steeply* (34k→68k: 0.066→0.234), so the ceiling is higher. Width 512
already saturates capacity (2048 adds nothing), so the loop's value architecture was fine —
it was simply trained on far too few samples each generation, producing the uninformative
leaf that collapsed PIMC to the passive prior.

**Fix (cheap, directly on the diagnosed cause): decouple value-training data from
per-generation search volume.** Value targets are just `(features, outcome)` — obtainable
from *no-search* master self-play (this sweep made 85k samples in 244 s). Only the *policy*
target (visit counts) needs the expensive search. So: pre-train + **accumulate** the value on
a large cheap self-play corpus (replay buffer across generations — the Decision-I window, now
clearly indicated) so the leaf carries real signal (R²~0.23+, higher with more data) *before*
it drives search, then re-run the loop. R²~0.23 is a categorically different regime from ≈0:
the leaf can credit positions, so search need not collapse to the prior. Open question before
committing loop compute: a **bigger data sweep** (150k–300k samples) to find where R²
plateaus — i.e. how much value headroom exists.

#### Real-corpus value ceiling — R² ≈ 0.44 (2026-06-04)

The self-play sweep ignored a far bigger, real-game corpus: **`materialised_full_v5/bc` holds
1.25 B play rows, each already carrying the 224-dim v5 feature *and* a `round_outcome` label**
(plus `skill_decile`). `scripts/value_ceiling_materialised.py` fits the value directly on it,
filtered to `skill_decile=9` (skill is a policy input, not a feature column, so mixing deciles
adds irreducible target noise); `round_outcome` is stored in raw score units and is /100'd to
the value target scale.

| h=512 | test R² |    | capacity @1.1M | test R² |
|---|---|---|---|---|
| 113k rows | 0.302 |    | h=512 | 0.437 |
| 340k rows | 0.390 |    | h=1024 | 0.439 |
| 1.13M rows | **0.437** |    | h=2048 | 0.441 |

**Test R² ≈ 0.44 — ~2× the self-play 0.23 — with a negligible train/test gap (0.445/0.437),
so it generalises.** Even 113k rows already gives 0.30. Capacity saturates at h=512 (the loop's
architecture was correct). **Triple-confirmed: the loop's R²≈0 was data starvation, not a
ceiling.** The value is highly learnable; the approach is not structurally capped.

**The fix, now concrete:** pre-train a `ValueBaseline(224, hidden=512)` on the decile-9
materialised corpus → an R²≈0.44 leaf available from generation 1 — no self-play needed for
the value. Persist via `save_critic` → `critic_0` for a re-run of the loop, and keep
training it on **accumulated** self-play (replay, Decision I) so it tracks the policy's drift
off the human-decile-9 distribution. R²≈0.44 is a categorically different leaf from ≈0, so the
re-run is the real test of whether a value with signal lets PIMC press the lead instead of
collapsing to the passive prior. (Necessary, not proven sufficient — the re-run decides.)

#### Cheap test result — a strong value does NOT fix passivity (2026-06-04)

Before the 12 h loop re-run, the cheap gate: pre-train the value on the materialised
decile-9 corpus (`scripts/pretrain_value_materialised.py`, 2.1 M rows → **held-out R²=0.459**),
plug it into **frozen** PIMC as the leaf (`configs/eval_behavioral_search_materialised.yaml`),
and run the deep behavioral dial (360 sims, 4 pooled slices, n=151 opportunities).

| deep frozen search, n≈40–151 | `caller_passivity_rate` |
|---|---|
| master | 0.183 |
| master-critic (R²≈0.11) | 0.323 |
| exploration-critic | 0.247 |
| **materialised-critic (R²≈0.46)** | **0.291** |

**Quadrupling value quality (R² 0.11 → 0.46) moved passivity ~nothing:** 0.291 vs the weak
critic's 0.323 is **−0.6σ (identical)**, and vs master 0.183 is **+2.2σ (still decisively
worse)**. **Value R² is not the lever for caller passivity.** Three value interventions —
frozen-weak, exploration-retrained, real-corpus-strong — all land 0.25–0.32, none below
master. So the decoupled-value loop would almost certainly not fix passivity either; the
cheap ~1 h test **saves the 12 h run.**

**Reframing.** The bottleneck is not value estimation. It is the **policy/prior** (seeds PUCT,
plays the in-tree opponents) and/or the **search structure** — or the *premise* is partly
wrong: if pressing the lead in these caller states is not actually +EV, a correct value
*should* decline to credit it and the "passivity pathology" is mis-measured. That root
assumption is now cheaply testable on the 1.25 B-row corpus: **in caller-following-with-legal-
beat states, do decile-9 experts press or cede, and does pressing yield a better
`round_outcome` than ceding?** That interrogates the premise before any further search/learning
engineering — and is the indicated next step instead of the loop re-run.

#### Premise test result — pressing the lead is NOT +EV (2026-06-04)

`ForcedPressAgent` (`search/forced_press.py`, registered `forced_press`) plays the master
everywhere except the exact states the telemetry flags as caller passivity, where it forces
the master's best **non-Pass** move; a test confirms it drives `caller_pass_events` to zero
through the real detection. Head-to-head vs master over the full 2000-deal pool (seat-swapped,
n=4000, `configs/eval_tournament_forced_press.yaml`):

> **forced_press vs master: mean delta = −1.4 (call-bonus −0.1), 95% CI [−8.9, +6.5].**

**Pressing the lead in caller-following states is not +EV** — the point estimate is slightly
*negative* and the CI comfortably spans zero, and the call bonus (the theorised mechanism)
is unmoved. **The "caller-passivity pathology" does not correspond to an EV loss: the master
cedes those tricks appropriately.**

This retroactively explains the whole arc. Every intervention "failed to reduce passivity"
because there was no EV on the table — a *correct* search declines to press a move that isn't
better, so deep search getting "more passive" was search correctly discovering that ceding is
fine, and a strong value (R²=0.46) didn't help because a good value shouldn't credit a
non-improving move. (Caveats: this forces the *best non-Pass*, so a more sophisticated press
might find EV the naive one misses; and it measures EV-vs-master, not vs an oracle. But the
simple "press more" hypothesis driving the line is falsified at n=4000.)

**Strategic consequence.** The caller-passivity diagnosis that motivated Phase-2 search
(ADR-0030/0031) is undercut. The PIMC + search+learning machinery is built, correct, and
reusable, but **the specific pathology it targeted is not where the strength gap lives.** The
indicated next step is *re-diagnosis*: establish whether the master is actually sub-master and,
if so, locate the real EV gap (a decision-tape / EV-delta audit over many decision types,
not just the caller-press one) **before** any further search/learning compute. Do not run the
decoupled-value loop — its premise is gone.

## Risks

1. **Throughput (existential).** Even S2 + multiprocessing on one CPU box may be too slow.
   Mitigations: cut data-gen sims, scale processes, accept weaker targets. Phase-1-first
   exists precisely so we don't pay S2's cost before knowing the signal is real.
2. **Env-model mismatch** (Decision E) biases targets; bounded, monitor.
3. **Vicious cycle** — policy trained on visit targets from a still-passive value could
   amplify passivity. Guard: true-outcome value target + Dirichlet/τ exploration; the
   per-generation passivity dial is the early-warning monitor.
4. **Generations-to-signal unknown** — time risk; Phase 1 caps the bet.

## Success criteria

Across generations, `caller_passivity_rate` drops **below master 0.162** *and* strength vs
master ≥ 0 (master-level), with the play head still legal and calls still calibrated.
**Phase-1 minimum viable signal:** passivity moves materially down over 2–3 reduced-budget
generations — proof the loop's direction is right and S2 is worth building.

## Consequences

This commits Phase 2 to a multi-week build whose hard part is *throughput engineering*, not
the learning algorithm (the algorithm is standard AlphaZero adapted to single-rooted PIMC).
It reuses essentially the entire existing stack — search, value trainer, export, eval — and
adds exactly one training primitive (soft-target KL) plus, conditionally, one engine
(batched-MCTS). The exploration-critic pilot already de-risked the load-bearing premise
(value responds to its training distribution) and ruled out the cheap value-only shortcut,
so this is the indicated — not speculative — next investment.
