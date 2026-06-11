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

## Build (2026-06-11, test-first, all green)

`ppo/vine.py` (`collect_vine_rows`, `vine_net_batch`) reusing the blunder-miner's
`record_round` / `playout_from` / `candidate_alternatives` and `MLAgent.from_loaded`
over the in-memory nets; `ParallelRollout.collect_vine` + `_vine_chunk` (workers reuse
loaded skeletons); `train_cotrain(vine_collect=)` replaces the play group post-GAE;
CLI `vine:` config block + `vine_rows` column/console dial. 86 ppo tests green
(4 new in `tests/training/ppo/test_vine.py`). Smoke on the real recovered 06225 nets:
warm-start, 256 vine rows/iter, bundle round-trip — clean. Run live in
`data/runs/cotrain_vine_v1`.
