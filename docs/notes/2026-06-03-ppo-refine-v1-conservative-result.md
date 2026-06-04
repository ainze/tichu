# 2026-06-03 — PPO Refine v1 (conservative config): dial moves −10%, Tournament flat. Ship gate NOT met.

## TL;DR

The first real PPO Refine run (warm-started from the full-corpus v5 `master` BC,
`step_e000_b01250000.bin`) **works mechanistically but is too weak to ship**. After
re-tuning out an early failure, the `caller_bomb_passivity_rate` dial fell from the
**0.323** master baseline to **~0.285 (−10%)** — but it **plateaued by ~iter 70**, and
a full-strength Tournament shows the sharpened policy is **statistically tied with
master** (`+2.1`, CI `[−4.74, +9.08]`). The behavioral fix did **not** translate into
wins. Per the ADR-0029 pre-committed kill logic this is "dials moved but
Tournament-vs-master stayed within noise" → **do not ship this checkpoint**; escalate
the levers or rethink scope.

## Run history

- **Run-1** (`kl.target=0.02`, `ent_coef=0.01`, `critic_lr=1e-3`): diverged from the
  goal. `value_loss` flat ~7000 (critic not learning), and entropy fell **while KL was
  reined back toward BC** → the policy collapsed back onto passive BC. Killed ~iter 20.
- **Run-2** (`kl.target=0.04`, `ent_coef=0.02`, `critic_lr=3e-3`): healthier — `value_loss`
  declined to a ~6000 floor, KL rode freely (~0.05, anchor at its `min_coef` floor), and
  the dial finally moved. This is the run the numbers below come from.

## Behavioral dials (self-play, 2000 seat-rounds, vs master baseline 0.323 / 0.170)

| metric | master | iter20 | iter70 | iter150 |
|---|---|---|---|---|
| **caller_bomb_passivity** | 0.323 | 0.328 | 0.296 | **0.285** |
| caller_passivity (broad) | 0.170 | 0.154 | 0.168 | 0.169 |
| bomb_when_legal | 0.104 | 0.114 | 0.110 | 0.115 |
| tichu_success | 0.774 | 0.737 | 0.770 | 0.752 |

Headline decelerates: −0.032 over iters 20→70, then only −0.011 over 70→150 (within the
~±0.026 SE of a small-denominator rate). **Plateau by ~iter 70 at ~0.29.** Over-correction
guard clean (`bomb_when_legal` barely moved). Broad passivity never moved.

## Tournament (full-strength, 2000 positions × seat-swap, bootstrap CI)

```
ppo_iter150 vs master:     +2.08   CI [−4.74, +9.08]   ← includes 0: NO win
ppo_iter150 vs bc_neutral: +20.69  CI [+14.67, +26.72] ← but master beats neutral by MORE (+23.2)
master      vs bc_neutral: +23.19  CI [+17.36, +28.75]
```

Sanity: decile-9 master > neutral (+23, as known); all ML ≫ rule (+203–212) ≫ random
(+270–273); rule > random (+154). Matrix calibrated; the `+2.1` vs master is noise.
Beating `bc_neutral` is purely the decile-9 conditioning effect — the **sharpening added
nothing on top of it**. Move-Prediction-Eval not run: irrelevant once the Tournament gate
fails.

## Interpretation

The mechanism is real — PPO did shift bomb-valuation — but a −10% reduction in a
**rare-event** error rate is worth only ~a couple points/round, which drowns in Tichu's
score variance. To beat master the dial needs to reach **single digits**, not −10%, and
the conservative config plateaus far short. Best read of the plateau: the decisive
bomb-caller spots are rare **and the opponent league doesn't punish passivity** (opponents
are timid clones of a barely-improved learner → no exploitation pressure → no gradient to
stop passing). The `value_loss` floor (~6000 ≈ RMSE 77) is largely **irreducible** variance
of `round_outcome`, not critic ignorance — the dial moved despite it — so the critic is not
the primary bottleneck (warm-up is a secondary lever, useful mainly to confirm the floor).

## Decision

Pre-committed fork (ADR-0029 §kill): **dials moved, Tournament within noise → don't ship.**
Two paths:

- **A) Escalate the same approach, decisively** — we only ever ran a *timid* config. One
  strong run: `ent_coef` 0.02→0.03 (huge over-correction headroom), a critic warm-up, and a
  fresher/larger league (`snapshot_every` 10→5, `max_snapshots` 5→8) so opponents skew toward
  the more-aggressive recent learner and actually punish passivity. Bet: the plateau is the
  weak-lever config, not the method's ceiling. (Set up as `configs/ppo_v5_escalate.yaml`.)
- **B) Question the v1 scope** — on-turn caller bomb-passivity may simply be too rare to move
  win-rate; the strength gap may need the ADR-0029-descoped pieces (out-of-turn bombs, the
  Game-to-1000 meta-game, Phase-2 search). If escalation A *also* moves the dial without
  moving the Tournament, B is the conclusion and we stop tuning PPO Refine.

Running A first.

## Escalation result (path A) — also fails; B confirmed

The escalation run (`configs/ppo_v5_escalate.yaml`: `ent_coef` 0.03, 15-iter critic
warm-up, league `snapshot_every` 5 / `max_snapshots` 8) completed 500 iterations. In-loop
it converged healthily (KL rode ~0.04 in-band, entropy held ~0.35, `value_loss` floored
~5000–5400 — the warm-up + critic_lr pulled it modestly below run-2's ~6000, confirming
~5k is near the **irreducible** floor). But the result is **worse on the dials and still
no win**:

| run | caller_bomb_passivity | caller_passivity | Tournament vs master |
|---|---|---|---|
| master baseline | 0.323 | 0.170 | — |
| conservative (run-2, iter150) | 0.285 (−10%) | 0.169 | +2.08, CI [−4.74, +9.08] |
| **escalation (iter500)** | **0.316 (−2%)** | **0.183** | **+3.96, CI [−3.32, +11.09]** |

The aggressive config moved `caller_bomb_passivity` *less* than the timid one and made broad
`caller_passivity` *worse*. Best read: `ent_coef`=0.03 kept the play distribution too diffuse
to **commit** to the decisive bomb — the entropy bonus fought the decisiveness the fix needs.
Tournament-vs-master CI still spans 0 (beating `bc_neutral` is again just the decile-9 effect;
`master` beats neutral by more).

**Conclusion: B.** The method was run at both ends of the lever range; neither beats master,
and pushing harder slightly hurt. On-turn-only PPO Refine v1 plateaus below a shippable win —
a −10% (best case) reduction in a rare-event error rate is worth ~2–4 pts/round, inside Tichu's
score variance. **Stop tuning PPO Refine v1.** The strength gap needs the ADR-0029-descoped
scope — out-of-turn bombs, the Game-to-1000 meta-game, or Phase-2 search — not more tuning of
the on-turn sharpening objective. No Sharpened Checkpoint ships; `master` remains the served
top tier.
