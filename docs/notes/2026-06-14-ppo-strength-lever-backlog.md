# 2026-06-14 — PPO / online strength-lever backlog (follow-up to the featurizer-v6 grilling)

## TL;DR

Captured during the featurizer-v6 session ([ADR-0038](../adr/0038-featurizer-v6-played-by-and-schupfen-received.md)) as the **strength** half, deferred behind the BC-base work. BC is label-bound, so these — *not* more actor input — are where post-BC strength lives. Each targets a **measured** PPO bottleneck (critic R² under-fit, the rollout-Q variance wall, the sub-behavioral decision-quality gap), not actor capacity. None are committed; all need their own grilling/ADR before build.

## The levers (ranked by how directly they hit a measured bottleneck)

1. **Enrich the asymmetric (Perfect-Info) critic on the high-variance lever decisions.**
   - *Targets:* the measured co-train critic under-fit — R² play 0.45 but **grand 0.10 / schupfen 0.25** (the lever decisions the critic is worst at).
   - *Idea:* the critic is train-time-only and may cheat freely; give it grand-tichu/schupfen-specific perfect-info signals beyond the (3×56) true-hand grid (e.g. oracle call-success-given-all-hands). Decided in the co-train/ADR-0034 line; the asymmetric-critic PPO smoke is the live decider post-[ADR-0033](../adr/0033-perfect-info-critic-escape.md).

2. **Potential-based reward shaping**, Φ = value-baseline (or `trick_points` capture).
   - *Targets:* the **rollout-Q variance wall** that drowned every advantage signal (ADR-0034/0035, piKL/pMCPA post-mortems).
   - *Idea:* policy-invariant shaping (preserves the optimum) for pure variance / credit-assignment reduction across the long round horizon. The `{trick_winner, trick_points}` step-info hook was built for exactly this and is untested as a shaping signal.

3. **Distill the Claim Solver + mined blunders into corrected labels** (endgame / blunder states).
   - *Targets:* the **sub-behavioral decision-quality** gap (owner-beats-master is within-normal-play, not a missing behavior).
   - *Idea:* inject **ground-truth optimal** play where it's computable, instead of imitation. Evidence-backed reopening bet from the blunder-miner (true blunders ~0.25/round @ +60; paired-world replay pierces PPO's noise floor → "mine→distill"). The Claim Solver (ADR-0032) supplies sound endgame certainties; the counterfactual miner supplies the mid-game ones. **Classification resolved (v6-A grilling):** the Claim-Solver signal belongs **here as a label/distillation, not as a BC input feature** — as an input it only helps to the extent decile-9 humans already act on the certainty (small gain by construction) and risks label-uncorrelated variance; as a label correction it is unambiguously valuable and label-independent. Cheap pre-check before building: fire the solver on a decile-9 held-out endgame slice — high human agreement → redundant (drop); low → exploitable headroom only a label can capture (confirms this lever). See [ADR-0038](../adr/0038-featurizer-v6-played-by-and-schupfen-received.md) "Decisions not taken".

4. **Targeted exploration / entropy on `bomb-when-legal` and the call decisions** (not global entropy).
   - *Targets:* the measured **bomb-undervaluation / caller-passivity** pathology (caller-passivity note, 2026-06-03).
   - *Idea:* concentrate exploration pressure on the specific lever decisions the behavioral telemetry flags, rather than diffusing it over the whole action space.

## Near-free win already on the board (no RL)

- **Lower the Call-Network decision threshold below 0.5.** `master` under-calls purely as an `argmax` (P>0.5) artifact vs a 14.5% human base rate — it calls like a decile-8 human ([2026-06-03-master-matches-decile9-human-behavior.md](2026-06-03-master-matches-decile9-human-behavior.md)). A threshold A/B is no-RL, near-zero cost. Cheapest strength lever in the repo.

## Graveyard (do not re-propose without dodging the named failure mode)

Per-round runtime adaptation (pMCPA, ADR-0036), inference-time anchored search (piKL, ADR-0037), naive re-anchoring (−23.59 via self-play co-drift), more AWR (flat), search+learning loop (ADR-0031, falsified), bigger actor trunk (label-bound). New levers must dodge the **τ-continuation / self-inconsistency** family and the **variance wall** — not walk back into them.
