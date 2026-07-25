# 2026-07-24 — Advantage-SNR probe: the critic's residual is NOT irreducible; it is fixable bias concentrated in call-hinge / late-round states

## TL;DR

Built and ran the **same-state stochastic K-replay variance probe**
(`scripts/probe_advantage_snr.py`) — the decisive-but-unbuilt measurement the
2026-06-18/19 critic diagnosis named when it declared the play critic "maxed,
residual irreducible". The verdict **reverses per-phase**:

1. **At round start the critic IS at its floor** (epistemic share 10.8% of MSE;
   aleatoric sd ≈ 198). The aggregate R² = 0.627 / "irreducible floor" read was an
   **aggregation artifact**: round-start-like states dominate return variance and
   masked the rest.
2. **In mid/late round the residual is majority FIXABLE error**: epistemic share
   49% [41,58] at open, 61% [48,72] at mid, **74% [65,82] at late** (n=571 states,
   K=16 replays each, exact training rollout distribution incl. {champion,bc}
   opponent alternation). The aleatoric floor collapses (sd 121 → 74 → 44 →
   33.8 once someone is out) but the critic's error stays ~90-100 RMS flat.
3. **The errors are symmetric, heavy-tailed, and concentrated where rounds are
   decided**: caller-live states carry epistemic 10.5k vs 4.3k for no-caller;
   the worst states are near-deterministic call/endgame outcomes missed by
   ±300 (e.g. V=+19.8 where 16/16 replays average +357.5 with sd 4.5; V=+429
   where truth is −5.6). The critic fails to *resolve the call outcome*, in both
   directions — it is not biased toward a team.
4. **With GAE λ=0.95 these are gradient BIAS, not just variance.** Future-state
   V errors enter the advantage as targets (δ terms), not baselines: each
   misvalued state j>t contributes γ(1−λ)(γλ)^{j−1−t}·ε to A_t — small for one
   state (~12 pts of a 300 error 5 steps ahead), but call-outcome
   misvaluations PERSIST until resolution, and a persistent error ε over the
   remaining m steps sums to ≈ (1−λ^m)·ε (≈40% of ±300 = ±120 at m=10).
   The policy is trained to chase critic illusions in exactly the
   decisive states, and the greedy gate then (correctly) refuses to promote the
   result. At λ=1, V is a pure baseline and cannot bias the gradient at all.
5. **Exploration is dead**: median p(sampled action) = 0.99999, p25 = 0.979
   across all recorded play decisions (caveat: includes single-legal-action
   states); play entropy 0.236 → 0.158 over the run. Whatever signal exists,
   the rollouts barely visit counterfactuals.

Together with the reproduced plateau facts (below), the unified diagnosis of the
strength plateau: **biased advantage estimation in call-hinge/endgame states +
near-zero counterfactual visitation, inside a trust region that is otherwise not
binding** (play_kl runs at 0.05–0.12 against a 0.12 target with coef at the
floor — the ball is used, pointed at illusions).

## The plateau, reproduced from logs (what any lever must explain)

- Wishfix gated run (`cotrain_v6_pbrs_resid_wish_gated_wishfix`, 22.4k iters,
  170 gate windows): 4 promotions (iters 768/2432/4608/10368, margins
  +7.7/+5.3/+5.1/+5.1 — the last three clearing CI_lo by ≈+0.3, i.e.
  winner's-curse territory), then **92 straight windows without promotion**.
  Pooled learner-vs-champion margin over the stall: **−0.48 ± 0.21** (the
  learner spent thousands of iters BELOW its own promoted snapshot — the +5.1
  read at 10368 was inflated), trending +0.26/1000 iters to ≈+1.5 late.
  Fresh deals every window (`1e9 + eval_idx·n`), so pooling is valid.
- vs the shipped champion (cpfix3328): −6.9 @ 4608 → −3.5 @ 10368 → −0.7 @
  17150 → **−2.4 @ 22450** — converged to ≈ parity-minus, never above.
- Learning rate collapse ≈ 50×: +7.7 in the first 768 iters vs ≈+2 in the last
  12,000. KL-loosening 0.08→0.12 @ ~18.5k did not change the arc.
- 2048×8 trunk negative control: same arc at 4× cost (−9.6 vs master @ 3450) —
  capacity is not the constraint (trunk-probe preregistration note).

## Method (why these numbers are trustworthy)

`scripts/probe_advantage_snr.py`, two stages, both under the **exact training
distribution**: live `_rollout_weights.pt` learner + critic, sampled
`BatchedCoTrainPolicy` on all four seats, opponents alternating
{champion, bc} as `train_cotrain` does, `learner_team=0`, pools seeded 9e6 / 9.5e6.
**Honesty caveat on "held-out":** training consumes pool seeds `0 + iter·512`
(one seed per position), so after 22.4k iters it covered seeds 0–11.5M — the
probe's deals were each seen ONCE, ~4–5k iterations ago (the `_diag` script's
"disjoint" comment was only true for short runs). Direction of bias:
conservative — a once-seen deal can only make the critic look better, so the
epistemic shares below are if anything understated. Mid-round STATES are never
literally revisited regardless (they depend on the sampled path). Future runs:
use `--pool-seed >= 100_000_000` (training reaches 100M only at iter 195k).

- **Stage A (round start):** 256 positions × 16 replicas through
  `collect_rollout`. Per position: E[R|s0], Var(R|s0), V at the first Schupfen
  step. Decomposition `MSE(V) = (V−E[R|s])² [epistemic] + Var(R|s) [aleatoric]`,
  with the −Var/K unbiased correction.
- **Stage B (mid-round):** 192 fresh rounds recorded one game at a time via a
  recording wrapper; ≤3 learner-seat Play states per round stratified by phase
  (open ≥11 / mid 6–10 / late ≤5 cards); each state K-replayed ×16 through the
  real driver (`_GameRun` + `_drive`) with everything sampled. `asked_tichu`
  at a resume is exact from the state (a seat has made a non-Pass Play iff it
  holds <14 cards). Outcomes team-0-relative from round start — identical to
  the trainer's reward. Bootstrap CIs over states.
- Reproducible: seeded torch Generator; ~15 min CPU total.
- **The probe is now a standing benchmark**: `advantage_snr_probe_v1c/
  stage_b_states.parquet` stores each state's 759-dim critic features +
  E[R|s]/Var(R|s) labels, so any candidate critic is scored in seconds with no
  replays (`epistemic share` per phase is the metric). Guard against the
  ADR-0033 lesson: this scores critics on-distribution against replay ground
  truth, not held-out R² of round-unique features.

## Numbers

| slice | n | aleatoric sd | epistemic var | critic MSE | epistemic share [95% CI] |
|---|---|---|---|---|---|
| round start (A) | 256 | 198 | 4,709 | 43,746 | **10.8%** |
| open (B) | 192 | 121 | 14,254 | 28,884 | 49% [41, 58] |
| mid (B) | 192 | 74 | 8,580 | 14,032 | 61% [48, 72] |
| late (B) | 187 | 44 | 5,679 | 7,647 | **74% [65, 82]** |
| caller-live (B) | 484 | 88 | 10,486 | 18,245 | 57% [51, 64] |
| no caller (B) | 87 | 73 | 4,264 | 9,651 | 44% [30, 57] |
| someone out (B) | 219 | 34 | 1,917 | 3,062 | 63% [53, 72] |

Deal-luck (between-position) variance at round start is only ≈14k of the ≈55k
total — the dominant term is continuation-sampling variance (39k), which is
exactly what deterministic paired playouts (miner/vine) cancel and what no
state baseline can.

Residual distribution is heavy-tailed (all-states resid² median 1.6k vs p90
27.7k): the critic is fine on typical states and catastrophically wrong on a
tail of call-hinge/endgame states with near-deterministic outcomes.

**Replications (both confirm within CI):**
- **Clean-seed** (`advantage_snr_probe_v2`, pool seed 100M — deals training
  NEVER saw): open 47% [38,55], mid 59% [49,69], late **71% [61,80]**,
  all 54% [49,60] — the seed-overlap caveat above is retired.
- **Shipped champion's critic** (`advantage_snr_probe_cpfix`, cpfix run
  weights, its own {champion,bc} opponents): open 55%, mid 68%, late
  **77% [63,88]**, all 64% [56,70], someone_out ale_sd 25 vs epistemic RMS
  ~45 (77%). Both lineages' critics share the same fixable blindness — this
  is a property of the recipe, not one run.

## What this changes in the closed ledger

- **"Critic network maxed / residual irreducible" (2026-06-18/19) is
  RETIRED** as stated. What remains true: capacity-as-architecture was null
  (typed critic), perfect-info input is marginal (ρ≈0.78), and the *aggregate*
  MSE is floor-dominated. What is new: the floor argument does not apply to
  mid/late/caller states, where 50–80% of the residual is fixable and the
  aleatoric floor (34–67 sd) sits BELOW the mined per-blunder signal (+60).
- The wish-loosening and play-escape negatives get a mechanism: more freedom =
  faster chasing of critic illusions (λ<1 bias), not just "more noise".
- Vine's safety (fixed-anchor, EV-neutral at 2.5% drift) and the CE-distill
  regression are both consistent: paired-evidence gradients don't chase
  illusions; hard labels fight calibration.
- Mechanism-level kills all SURVIVE the 2026-06/07 bug-fix wave (team_scores
  was serve-only — tournaments are single rounds from 0-0; schupfen-cpfix and
  wishfix are already embodied in the current lineage; trunk capacity was
  re-tested at v6). Audited this session; nothing there to reopen.

## Hygiene finding (separate)

The `ppo.pbrs:` block in `cotrain_v6_pbrs*` configs is **silently ignored** —
PBRS shaping never merged into `src` (`git log --all -S pbrs -- src/` is
empty). The whole "pbrs"-named lineage (incl. the shipped champion) trained
unshaped. No strength impact (PBRS measured advantage-neutral/dead-tie), but
the configs lie; remove the block or land the code.

## Pre-registered next test (the lever decision)

The cheapest falsification of the diagnosis, in order:

1. **λ-bias arm (near-free):** resume the wishfix run (or fork fresh from its
   champion) with `ppo.lam: 1.0`, everything else byte-identical. At λ=1 the
   critic cannot bias the gradient (pure baseline). Read: greedy-gate windows —
   does the champion axis wake up (promotions with CI_lo>0 at honest margins)?
   Bar: ≥2 promotions in the first ~40 windows whose pooled post-promotion
   margin stays >0 (no winner's-curse giveback), or CI-above the old run's
   pooled −0.48 stall band over the same window count.
2. **Critic-stratum arm:** aleatoric-normalized value loss (weight each step's
   MSE by 1/Var̂(R|phase) or a per-phase scale head), validated FIRST on the
   standing benchmark (epistemic share late < ~40% would be a hit), then the
   same gate read. Auxiliary exact labels (claim-solver certainties,
   call-outcome heads) are the escalation.

   **RAN 2026-07-25 (`scripts/critic_fix_offline.py`, `data/runs/critic_fix_v1`)
   — NEGATIVE for loss reallocation.** 229k rows collected under the frozen
   λ=1-arm weights ({champion,bc} alternation); three arms fine-tuned from the
   incumbent critic (scratch training is ~1000× data-starved vs the incumbent
   and only measures under-training): plain-MSE control, aleatoric-normalized
   MSE (late states ~8× mean weight), heteroscedastic NLL (learned σ(s)). All
   three converge to the incumbent's fit: benchmark late share 0.72→0.71 (v1c),
   0.70→0.69 (v2); caller-live 0.57→0.56. Bar (<0.40) missed by a mile.
   **Interpretation: the late/caller epistemic error is a REPRESENTATION
   limit, not an allocation artifact** — no reweighting of the same loss on
   the same 759-dim input touches it; the ±300 misses are on near-deterministic
   combinatorial positions (call fulfilment, guaranteed outs) that an MLP
   cannot card-count. Consistent with the typed-critic null.
   **Remaining live branch of lever 2:** oracle/computed features as
   train-time critic input — run the Claim Solver (+ oracle call-success
   indicators) on all four hands per state and feed the certainty bits to the
   critic (it may cheat freely), e.g. as a frozen-incumbent + small additive
   correction head. Pre-check first: regenerate benchmark states (seeds are
   deterministic) capturing solver fire/verdict per state — if the solver does
   not cover the heavy-tail miss states, lever 2 is DEAD and lever 3
   (frozen-reference vine + ratchet) is the funded bet.

   **COVERAGE PRE-CHECK RAN 2026-07-25 — LEVER 2 IS DEAD.**
   (`probe_advantage_snr.py --solver-capture`, fresh self-consistent benchmark
   `advantage_snr_probe_v3`: frozen λ=1-arm weights, pool seed 120M, n=572;
   shares replicate a third time: 0.45/0.62/0.67.) Per-seat chain
   Guaranteed-Out (`go`) + Reclaim (`rc`) bits vs the residual tail:
   - **Zero enrichment**: any_go fires on 37.1% of tail states (|resid|≥150)
     vs 42.9% of NON-tail states; reclaim 46.8% vs 56.1%. The critic's worst
     misses are not where the solver's certainties live.
   - Direction-aligned coverage (a guaranteed out on the side V underrated):
     29/125 = 23% of the |resid|≥100 tail — the ceiling on what any
     solver-feature correction could touch.
   - 8 of the worst-10 misses carry NO solver bit; several have replay sd
     0–15 with no guarantee anywhere: their outcomes are
     **policy-deterministic** (near-argmax play by all four seats fixes the
     future), not **rules-deterministic**. No sound solver can label those —
     only playouts can. That is the same information paired deterministic
     playouts produce exactly, which is why the estimator family (vine) sees
     what no critic input can.
   **Verdict: the critic cannot be fixed by loss, capacity, type-conditioning,
   perfect-info inputs (all previously null) OR computed-certainty inputs
   (this check). The epistemic error is irreducible *for a state-function*
   even though it is not irreducible in principle — the missing information
   is "what the current policies will actually do", i.e. a playout.
   Lever 3 (frozen-reference vine + greedy-gated ratchet) is the funded bet.**
3. **If both fail** → the structural bypass: frozen-reference vine (paired
   deterministic advantages for play; branch rows double as targeted
   exploration) + the greedy-gated ratchet for anchor advancement — the two
   halves that were never combined (vine died 06-12/13 before the greedy gate
   existed 06-19, and its re-anchoring death was the self-play-continuation
   co-drift the frozen reference removes by construction).

## Blunder-pool refresh on cpfix3328 (ran this session — the signal budget)

`data/runs/blunder_mining_cpfix3328`: 1,500 rounds tier-1 (149,447 rows;
hindsight-positive profile ≈ identical to v1's old-champion mine — best-alt
≥+15 at 10.4/round vs 11.1), then the v1 stratified tier-2 protocol (150/band
at 24 worlds + EV-neutral control):

| band | survival | robust/round | survivor world-Δ |
|---|---|---|---|
| control \|δ\|≤5 | **0/150 = 0%** | — | (v1 FP control: 0.7%) |
| [15,50) | 5.3% | 0.158 | +108 |
| [50,150) | 3.3% | 0.128 | +96 |
| [150,400) | 1.3% | 0.039 | +126 |

**Total ≈ 0.32 robust blunders/round, raw pool ≈ +34/round** (upper bound —
selection-on-24-worlds inflates; v1's 96-world re-verify deflated per-case
magnitudes ~2×, so expect ~+15–20 deflated). Survivors are again rule-less
(max context cluster n=2). Verdict: **the sub-noise EV pool persisted through
the +8.8 champion upgrade and all three bug fixes** — the shipped progress
came from elsewhere (schupfen/wish fixes, coarse play gains), not from
draining this pool. The pool remains the payoff budget for whichever lever
finally delivers a clean per-decision signal.

Explicitly NOT reopened: leash loosening (measured harmful twice — and now
mechanistically explained), trunk capacity, PBRS/potential shaping
(advantage-neutral math), pMCPA/piKL/search-family (self-inconsistent
continuation targets).

## Artifacts

`data/runs/advantage_snr_probe_v1/` (stage A+B, first pass),
`advantage_snr_probe_v1b/` (stage B, 571 states + descriptors + CIs),
`advantage_snr_probe_v1c/` (same states + 759-dim critic features = the
standing critic benchmark), each with `summary.json` + `stage_b_states.parquet`.
Rerun: `py scripts/probe_advantage_snr.py --config
configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml --run-dir <run> --out-dir
<out> [--stage a|b|both]`.
