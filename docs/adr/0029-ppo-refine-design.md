# PPO Refine design

We will sharpen the BC Checkpoint with **PPO Refine** — online self-play through
the rules engine, warm-started from BC and KL-anchored to the frozen BC policy —
to fix the diagnosed bomb / strong-combo undervaluation (caller passivity; see
[docs/notes/2026-06-03-caller-passivity-bomb-undervaluation.md]). The diagnosis is
closed and offline methods are exhausted (BC is label-bound; AWR came back flat —
[docs/notes/2026-05-29-awr-game-target-flat.md]). This ADR records the full design
locked in the design interview; **PPO Refine** / **Sharpened Checkpoint** are
glossed in [CONTEXT.md](../../CONTEXT.md) §"Phase 2 / online-RL terms".

## Decisions

**Algorithm.** PPO actor-critic, warm-started from the BC Checkpoint, with a
**KL-anchor to the frozen BC policy** + an entropy bonus. Chosen over DouZero-style
DMC (we are sharpening an already-strong policy on single-machine compute, not
learning from scratch at cluster scale) and over AlphaZero-style search (Phase 2).

**Reward & episode.** Reward is **pure-terminal `round_outcome`** (team-relative,
**includes the ±100/±200 call bonus**) and an **episode is one Round** (deal →
`_finalise_round`). The call bonus is non-negotiable: caller passivity is only
"wrong" because passing forfeits the bonus. **No trick-point reward shaping** even
though the engine `step` `info` exposes `trick_points` — shaping toward immediate
card-points rewards exactly the immediate-capture-over-terminal-bonus behavior that
*is* the pathology (the decisive caller-passivity tricks are often 0-point). Credit
assignment over the sparse terminal reward is the critic's job (GAE), not a
hand-crafted proxy's. The Game-to-1000 meta-game (e.g. desperate calls when
trailing) is **out of scope for v1**.

**Scope of the update.** PPO optimizes the **Trunk + play head only**. Calls and
Schupfen are separate networks and stay frozen (the diagnosis says they are fine).
Wish and dragon_assignment **share the Trunk** with play, so they cannot be held
behaviorally fixed while the Trunk moves — therefore the **KL anchor is applied to
the entire Policy Network** (play + wish + dragon) against frozen BC, started
**tight** on wish/dragon, rather than "freezing" head weights that would drift
anyway. Self-play conditions **all seats on a fixed Skill Decile 9** (master is the
strength goal; conditioning lower deciles would teach weaker play). Because the
Trunk is sharpened at decile-9, the Sharpened Checkpoint **ships master-only** — it
is **not** reused under the ADR-0027 decile-mapped serving trick; medium/hard keep
serving the BC Checkpoint.

**Out-of-turn bombs — on-turn only for v1.** The engine supports `BombInterrupt`,
but (a) the entire diagnosed failure is on-turn (the `caller_passivity` probe only
fires on the caller's own turn; on-turn bomb deployment is already a normal Intent),
and (b) nothing in the `Agent` / inference path solicits out-of-turn bombs, so a
learned interrupt could not ship without new solicitation paths through the Agent
ABC, rollout, and serving codec. Train and serve therefore match (neither
interrupts) — the correct environment for a *sharpening* objective. Known blind
spot: interrupt-exploitability won't surface in the caller-passivity dials; it needs
a dedicated probe if revisited.

**Throughput — single-process vectorized self-play.** Hold M concurrent games
(start M≈256) as plain `GameState`s in one process. Each tick: gather
current-player `PrivateState`s, `featurize` and stack, **group by serving model**
(learner / frozen-BC / league-snapshot) and run **one batched forward per group** on
the GPU, sample legal-masked Intents, record `logπ`/value, resolve to ConcreteAction
via the existing Resolver, and `step` all engines. Record trajectories **only for
the two learner-team seats** (parameter-shared partner → both on-policy; opponents
are environment dynamics). The learner and actor **share the in-memory grad-capable
`BCModel`**, so trajectories never cross a process boundary. This deliberately
rejects the Tournament's process-parallel **batch-1** pattern (ADR-0026) for the
rollout — that batch-1 / IPC shape is what made AWR throughput painful. Do **not**
reuse `play_full_round` / `MLAgent.act` (batch-1, argmax, TorchScript-export); they
stay for eval. The non-batchable residue (`featurize`, legal-mask build, Resolver,
engine `step` — all per-game Python) scales with M; **profile before optimizing it**
(a CPU actor pool is the escalation, not the starting point).

**Opponent / partner structure.** Parameter-shared partner (both learner-team seats
run the learning policy; the seat-invariant featurizer makes this clean) + opponents
sampled from a **small league** (frozen BC at the start, plus periodic snapshots of
the improving learner). Self-play surfaces bomb/caller situations far more than the
rare-in-corpus BC data, and the league forces opponents to eventually respect bombs.

**Value function — separate critic off the Trunk.** The critic is its own MLP over
the 224 features (extend `tichu_training/awr/value_baseline.py`), fit online on GAE
returns. It is kept **off the policy Trunk** so the KL-anchored representation is
shaped only by policy gradient + KL-to-BC + entropy — a value head sharing the Trunk
would inject value-loss gradients that fight the anchor. (The AWR ~14%-variance
value fit does not transfer: that was corpus returns where bombs are rare; the PPO
critic fits self-play returns where bomb/caller situations are common.) Fallback if
early-training advantage noise hurts: a **stop-gradient** shared-Trunk value head
(`value_head(trunk(x).detach())`) — reuses the representation without perturbing the
Trunk. A short critic warm-up under the frozen policy is the other mitigation.

**Hyperparameters (conservative starts).** Two distinct KLs: clip ε = **0.1** bounds
the trust-region step (old↔new); **β_KL** penalizes anchor KL (learner↔frozen BC) and
is **adaptive** to a small KL-to-BC budget. γ = **1.0** (short episodes, terminal
reward), GAE λ = **0.95**, entropy ≈ **0.005–0.01**, policy LR **1e-4**, critic LR
**1e-3**, **2–3** epochs/batch, **~256–1024** rounds/update, advantage-norm **on**.
**β_KL and entropy are the two active levers** (a tension pair: the anchor pulls
toward passive BC, entropy+gradient push toward bomb deployment); everything else is
set-and-leave.

**Eval / stopping / kill.** Behavioral dials (`caller_passivity_rate`,
`caller_bomb_passivity_rate`, `bomb_when_legal_rate`, call-success) and
Move-Prediction-Eval run **every snapshot**; the **Tournament Matrix** runs at
milestones against the **full zoo at once** (frozen BC, master, AWR-Refined, several
PPO snapshots). **Ship** when passivity falls hard, Tournament beats master with CI
excluding 0, Move-Prediction-Eval play top-1 drop is bounded (pre-commit ≤ ~3pp),
and the owner play-test passes. **Kill** (AWR-style, pre-committed) when: a committed
compute budget is exhausted with bomb-passivity not meaningfully moved; or the dials
move but Tournament-vs-master stays within noise (fix didn't translate to wins); or
no β_KL/entropy setting satisfies both passivity↓ and Move-Pred-bounded. Commit now
to the rule **"if the dials haven't moved by 50% of budget, stop and diagnose rather
than extend"**; set the budget *number* after the rollout harness gives a rounds/hour
figure.

**Kill-criterion specifics (resolved 2026-06-03, run session).** Budget unit is
**iterations** (what the config controls and snapshots key to), with the implied
wall-clock as a backstop ceiling; the absolute number is filled in after the first
throughput read. Headline kill metric is **`caller_bomb_passivity_rate`** (baseline
**0.323**), not the broader `caller_passivity_rate`. "Moved meaningfully" = a **≥30%
relative drop by the half-budget snapshot** (≤ ~0.226) — a deliberately modest
direction-check, not the ship bar of single digits. Below it at halfway → stop and
diagnose. Dials are read by **exporting a persisted snapshot and running `eval_matrix
--mode behavioral`** (every snapshot for the first ~3 to confirm direction, then every
2nd–3rd). Enabled by mid-run observability added to `train_ppo`: per-iteration stats →
`{run_dir}/ppo_log.csv`; intermediate Checkpoints → `{run_dir}/snapshots/iter_NNNNN.bin`
every `snapshot_every`. Two self-play guards: **non-transitivity** — the latest snapshot must beat
frozen BC and master, not just recent predecessors (the full-zoo Tournament detects
RPS cycling; the owner play-test is the out-of-distribution backstop); and
**over-correction** — `bomb_when_legal_rate` must rise from BC's near-zero but stay
bounded, not → 1.0 (a bomb-spammer is a regression even if short-term Tournament
looks fine).

## Consequences

- New **vectorized self-play rollout harness** is the first build and the throughput
  gate; build it test-first before the PPO update loop.
- Wish/dragon are protected by anchor tightness, not by freezing — their
  Move-Prediction-Eval rows are a watched regression signal during training.
- The Sharpened Checkpoint stays byte-compatible with BC/Refined Checkpoints (same
  Trunk+Heads payload), so it remains a drop-in `master`-tier swap.
- v1 explicitly excludes: out-of-turn bombs, the Game-to-1000 meta-game, and
  decile-mapped serving of the Sharpened Checkpoint.

## Run result (2026-06-03, conservative config)

First real run (full-corpus v5 `master` warm-start). After re-tuning out an early
collapse, `caller_bomb_passivity_rate` fell 0.323 → **~0.285 (−10%)** but **plateaued
by ~iter 70**, and the full-strength **Tournament-vs-master came back +2.1, CI
[−4.74, +9.08] — within noise**. The behavioral fix did **not** translate into wins;
beating `bc_neutral` (+20.7) is purely the decile-9 effect (`master` beats neutral by
more, +23.2). Per the kill logic above ("dials move but Tournament-vs-master stays
within noise") this checkpoint **does not ship**. The `value_loss` ~6000 floor is
largely irreducible `round_outcome` variance, not critic ignorance (the dial moved
regardless), so the critic is not the primary bottleneck; the suspected ceiling is
rare-event frequency + a league that doesn't punish passivity. Escalation (stronger
entropy + critic warm-up + fresher/larger league, `configs/ppo_v5_escalate.yaml`) was
run next.

**Escalation result — method abandoned at v1 scope.** The aggressive config (500 iters)
moved the dial *less* (caller_bomb_passivity 0.316, −2%) and broad passivity *worse*
(0.183), with Tournament-vs-master still within noise (+4.0, CI [−3.32, +11.09]). Run at
both ends of the lever range, **PPO Refine v1 never beats master**; pushing harder slightly
hurt (high entropy kept the policy too diffuse to commit to the decisive bomb). **Decision:
stop tuning PPO Refine v1 — the on-turn-only scope is too narrow to move win-rate; no
Sharpened Checkpoint ships and `master` stays the served top tier.** The strength gap needs
the descoped scope (out-of-turn bombs, Game-to-1000 meta-game, or Phase-2 search). Full
write-up: [docs/notes/2026-06-03-ppo-refine-v1-conservative-result.md].
