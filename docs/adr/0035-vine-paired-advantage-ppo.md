---
status: proposed
---

# Vine / paired-advantage PPO — give the play head the luck-free signal PPO never had

The ADR-0034 closure attributed the +8.37 plateau to advantage noise: `R − V(s)`
carries the round's irreducible outcome variance (~55–60% even with the perfect-info
critic), drowning per-decision EV differences. Every lever that re-ran into that noise
(league, leashes, critic variants, more iterations) read flat. Two experiments on
2026-06-11 then verified, independently, both premises of a different estimator:

1. **The blunder-miner** (`docs/notes/2026-06-11-blunder-mining-counterfactual-replay.md`):
   comparing actions by playing both out deterministically IN THE SAME WORLD cancels
   deal-luck exactly — 51 true blunders confirmed at 96 worlds against a 0.7%
   false-positive control. The signal `R − V` cannot see is plainly visible to paired
   replay (order +10–15 pts/round below PPO's noise floor).
2. **The distill premise test** (`docs/notes/2026-06-11-distill-premise-test.md`):
   that signal has learnable structure (20.6% held-out fix-rate) but must NOT be
   delivered as hard CE labels — naive distillation regressed −18 h2h. It must arrive
   inside the policy's trust region.

This ADR records the build that delivers it as PPO advantages. Owner directive
2026-06-11: "try the untested training mechanism" — design choices locked by the
implementing agent, conservative defaults, kill bar below.

## Decisions

**Estimator — exact within-world paired advantages from dedicated vine games.** Each
iteration plays `vine.games_per_iter` (64) dedicated rounds with the CURRENT nets in
**argmax mode** (the policy as served; partner-trick guard off — we train the net). At
`decisions_per_game` (4) sampled play decisions with >1 legal action, the top-3
alternatives by the policy's own ranking are forced and played out deterministically
(`playout_from` — the blunder-miner's parity-tested resume). The chosen branch needs
no playout: with deterministic agents its continuation IS the recorded game (parity
invariant). The advantage is

    A(s, chosen) = R_det(chosen) − mean(R_det over all 4 branches),

team-relative to the actor, exact given the world (zero estimator variance), luck-free
across branches (same deal). Sampling worlds happens implicitly across vine games.

**Scope — play head only; everything else keeps ADR-0034's machinery.** Vine steps
REPLACE the play head's GAE group in the `CoTrainBatch` (~256 exact steps/iter instead
of ~15k noisy ones — the premise is precisely that fewer high-SNR samples beat many
noisy ones). Schupfen / calls / wish keep GAE on the main stochastic rollout; the
shared Perfect-Info Critic keeps training on all main-rollout steps (it still baselines
the other heads; play no longer consumes it).

**Off-policy containment.** Branch actions are argmax choices, not samples; `old_logp`
is the current policy's masked log-prob of that argmax action (ratio starts at 1), and
the standard clip (0.1) + per-net adaptive KL bound the mismatch — the same containment
that holds the rest of the stack. Per-net advantage normalization handles scale.

**Warm-start AND anchor = the shipped iter_06225 policy** (recovered exactly from its
TorchScript export — the raw snapshot was pruned; 0 missing / 0 unexpected keys). This
is a REFINEMENT run: `kl.play target 0.01` (tighter than the 0.02 used from BC) keeps
the policy at the +8.37 optimum while the vine gradient spends the budget only where
paired evidence disagrees. Rationale: the wish-loosening A/B showed free wandering
corrupts the trunk; the distill test showed even 0.9% uncontrolled drift costs −18.

**Vine positions from a disjoint seed stream** (`pool_seed 555000`) so play data is
uncorrelated with the main rollout's deals.

**Cost.** 64 games × (1 record + 12 branch playouts) ≈ 830 playout-equivalents/iter on
the worker pool ≈ +3–5 s on the ~7 s ADR-0034 iteration (measured ~12 s steady).

**Ship / kill bar.** Same as ADR-0034: seat-swap head-to-head vs iter_06225
(`configs/cotrain_vine_v1_vs_iter06225.yaml`), 95% CI > 0 to ship; flat after a
multi-thousand-iteration stretch at matched power → the paired-advantage lever joins
the closed ledger and the strength program re-closes. No auto-kill; Resume Bundle
every iteration; `check_cotrain` out of loop (OOM, ADR-0034).

## Consequences

- The play head's gradient no longer depends on the critic at all — critic quality
  questions (R² ceilings, rare-state generalization) are moot for play.
- The estimator is policy-relative (branches continue with the current argmax policy):
  it sharpens against the policy's own best understanding, the same regime as
  self-play. Multi-step plans the policy would never follow remain invisible.
- 256 steps/iter is a deliberate data-diet; if learning stalls from sample starvation
  (rather than signal absence), `games_per_iter` / `decisions_per_game` are the dials —
  cost scales linearly.
- Known smaller bias: argmax branch evaluation under a stochastic-trained policy, and
  exploration for play now comes only from branch comparison (entropy bonus retained).

## Addendum (2026-06-12): v1 verdict — delivery failure, not refuted premise; v2 enriches

**Strength reads (h2h vs iter_06225, seat-swap):** iter 500: +0.13 [−4.07, +4.40]
(n=8k); iter 1275: +1.67 [−0.37, +3.70] (n=40k); iter 2400: **+0.45 [−1.68, +2.46]
(n=40k) — flat at full power**; the 1275 read was noise upside.

**Autopsy** (`data/runs/cotrain_vine_v1/autopsy_fixrate.py`, scoring the export
against the 663 mined corrections — held-out w.r.t. vine's disjoint seed stream):
iter_2400 fixes **2.1%** of them (06225 baseline 0.2%) at 2.5% ordinary-decision
drift. Self-consistent: 2% of the ~+10–15/round mineable signal ≈ +0.2–0.3/round,
exactly the observed flatness. So the kill bar's premise — that the mechanism
delivered its signal and the signal did nothing — does not hold; the mechanism
delivered ~10% of its signal. Two structural causes:

1. **Three quarters of the computed evidence was discarded.** Only the chosen
   branch became a row; the alternative branches' exact returns fed the baseline
   and were thrown away. The discarded rows carry the corrective direction: at a
   blunder, pushing UP the better alternative directly, rather than pushing the
   chosen action down and letting the mass renormalize blindly.
2. **Near-tie dilution.** Blunders are ~1% of uniformly sampled decisions; the
   other rows have near-zero advantages that only feed the per-net normalization.

**Positive finding worth keeping:** trust-region containment held at 2.5% drift
(EV-neutral) where CE-distillation regressed −18 at 0.9% drift — paired-evidence-
backed drift is harmless. The mechanism is safe; v1 was just slow.

**v2 (`configs/cotrain_vine_v2.yaml`):** same playout budget; `emit_branches`
turns every branch into a row (≤1024/iter instead of 256), `min_abs_advantage: 5`
drops near-ties. Warm-start and anchor = vine-v1 iter_02400 (EV-equal to 06225,
keeps the 2.1%). The corrections fix-rate is the cheap mid-run progress dial
(minutes, no tournament): if it climbs toward ~15–20%, the +2–3/round becomes
measurable at n=40k; if branch-enriched delivery can't move it either, THAT is
the clean kill — the bar itself is unchanged (h2h CI > 0 vs iter_06225 via
`configs/cotrain_vine_v2_vs_iter06225.yaml`).

### Trust-region saturation (2026-06-12, the second binding constraint)

v2's first dial readings exposed the real geometry: the KL penalty is
`kl_anchor_loss` to the FROZEN warm-start anchor — a fixed-radius ball capping
TOTAL movement, not a per-step speed limit. v2 filled the ball in ~20 iterations
(fix-rate 4.8% at iter 25, play KL 0.025 > target), the adaptive coef slammed
0.002 → 32 and dragged the policy back (fix-rate 4.1%, drift reverting
94.7% → 96.7% agreement by iter 75). This retro-explains v1: its 2,400
iterations orbited a 0.01-ball around 06225 that holds ~2% of fixes; v2's
enrichment packs ~2.3× more correction into the same ball — instantly — and then
also parks on the boundary.

**Decision: `reanchor_play_every: 100`** — the play anchor becomes the current
play net every 100 iterations, converting the ball into a trail of contained
steps. Each step is the unit v1/v2 proved EV-safe twice (paired-evidence-backed
drift at one ball's distance read EV-neutral, where CE-distill's smaller drift
read −18). The wish head rides the play net, so its anchor moves too —
accepted, since wish's own GAE gradient stays leashed at 0.008 per step.
Honest caveat: EV-neutrality was measured at ONE ball's distance; compounding
steps is the new, unverified part. Guard rails: the fix-rate/drift dial
(`data/runs/cotrain_vine_v2/autopsy_fixrate.py`) every few hundred iterations
and the h2h read before believing anything. The Resume Bundle now carries the
moved play anchor (`play_anchor` field) — without it a restart would snap the
anchor back to the warm start.

## Build (2026-06-11, test-first, all green)

`ppo/vine.py` (`collect_vine_rows`, `vine_net_batch`) reusing the blunder-miner's
`record_round` / `playout_from` / `candidate_alternatives` and `MLAgent.from_loaded`
over the in-memory nets; `ParallelRollout.collect_vine` + `_vine_chunk` (workers reuse
loaded skeletons); `train_cotrain(vine_collect=)` replaces the play group post-GAE;
CLI `vine:` config block + `vine_rows` column/console dial. 86 ppo tests green
(4 new in `tests/training/ppo/test_vine.py`). Smoke on the real recovered 06225 nets:
warm-start, 256 vine rows/iter, bundle round-trip — clean. Run live in
`data/runs/cotrain_vine_v1`.
