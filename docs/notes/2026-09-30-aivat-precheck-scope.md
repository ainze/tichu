# AIVAT pre-check — scope (2026-09-30)

**Question.** Can an AIVAT-style luck correction (Burch et al., AAAI 2018) shrink
the CI of our A/B harness, so every future ship/kill test needs fewer Rounds?

**Short answer from reading the code: not in the paired self-play harness.**
Classical AIVAT is ≈ the duplicate seat-swap we already run. The pre-check
below is re-scoped to (a) confirm that cheaply, (b) test the one variance
lever that survives the argument (multi-fidelity control variates), and
(c) measure the deal-luck share — which is where AIVAT *does* apply (unpaired
live / human games) and which answers the original luck-vs-skill question.

## 1. Why classical AIVAT adds ~nothing over seat-swap here

AIVAT subtracts zero-mean terms at two kinds of node:

| node | term | in our harness |
|---|---|---|
| chance | `V(after) − E_outcomes[V]` | only chance event is the **deal**; the full 14-card deal is fixed per Position (`full_position_pool.py`) |
| action by known policy π | `V(s,a) − Σ π(a′)V(s,a′)` | **zero**: served `MLAgent` is greedy/argmax on every head (`ml_agent.py` — masked-greedy, `max(...)`, call threshold 0.5) |

With an agent-independent V, the deal term for team A in the seat-swapped half
is the exact negative of the first half's term (seat symmetry), so the terms
**cancel in the paired delta**. Pairing already removes the deal's main effect
*exactly as these agents experience it*; AIVAT would only remove an *estimate*
of it. Supporting evidence: A==B gives delta ≡ 0 (`tie_rate` in
`tournament.py`), and the partner-trick-guard A/B had 1,992/2,000 deals at
exactly 0.

What is left after pairing (~57% of unpaired variance, given the measured
seat-swap ρ ≈ −0.43) is the **deal × policy interaction**. It is deterministic
given the deal. No zero-mean chance/action correction can touch it.

**One real AIVAT term survives:** the 8/6 split. If the two halves' Grand-Tichu
calls differ, the value of the last 6 cards is revealed on different
post-call states. The term `V(s₁₄) − E_{6-card completions}[V(s₁₄)]` then no
longer cancels. It is expected to be tiny (it only exists on GT-divergent
Positions). M2 measures it.

## 2. What the pre-check measures

One instrumented paired run. No training, no new model.

- **Pair:** v7 champion iter-15360 vs its v7 BC (the drift-benchmark setup,
  `configs/drift_v7_iter15360.yaml`). This pair diverges on most deals, the
  opposite of a near-identical A/B.
- **Pool:** existing full-strength Position pool, n = 4,000 Positions × 2 halves.
- **V:** the v7 Perfect-Info Critic from `data/runs/cotrain_v7_gated/_rollout_weights.pt`,
  **copied first** because the live run overwrites it. Input:
  `featurize_perfect_info(game_state, seat)`. Bias in V does not matter
  (control variates stay unbiased). Only correlation matters.
- **Hook:** `play_full_round(..., decision_observer=...)` already hands over
  the full GameState at every one of the 6 Decisions. Log per half: the action
  sequence, the GT callers, the NN-call count, and V (team-A perspective) at
  checkpoints {first decision, post-GT, post-Schupfen, end of trick 1/3/5/8},
  plus the final margin and call bonus.

| id | measures | how | pre-registered bar |
|---|---|---|---|
| **M1** deal-luck share | fraction of single-game margin variance explained by the deal as a reference agent sees it | `ρ²(V_deal, M)` on unpaired halves; compare with pairing's 43% | none; this is a descriptive number. ≥ ~0.4 ⇒ AIVAT on unpaired live/human logs roughly matches seat-swap → follow-up (§4) |
| **M2** AIVAT over pairing | variance left after adding the GT-split term | `Var(d − Σ GT terms)/Var(d)`; K = 16 MC completions of the undealt 24 cards, **only** on GT-divergent Positions. Report that fraction first. | ≥ 0.90 ⇒ classical AIVAT closed for self-play A/B |
| **M3** multi-fidelity | cheap truncated-play proxy `d̃_t` (V at checkpoint t, both halves) as a control variate whose mean is estimated on many more cheap Positions (MFMC, Peherstorfer et al. 2016; unbiased) | per t: `ρ_t = corr(d, d̃_t)`, `c_t` = NN-call fraction up to t; equal-compute variance ratio `R_t = (√(1−ρ_t²) + √(c_t ρ_t²))²` | **build iff min_t R_t ≤ 0.60** (≥ 1.67× fewer Rounds at equal compute). Needs roughly ρ ≥ 0.85 at c ≤ 0.1. Prior: unlikely, since divergence snowballs after most checkpoints |
| **M4** post-stratification | free: strata on cheap deal features (bomb count, GT-grade hands, Dragon/Phoenix split, hand-strength spread) | fit strata on half the Positions, score on the other half | ≤ 0.80 ⇒ add stratified pools to the harness |

Also report `first_divergence_ply` per Position. It gives the NN cost the
lockstep prefix shares and puts a number on the "chaos after first divergence"
story behind the 57% residual.

**Decision rule.** M2 ≥ 0.9, M3 > 0.6 and M4 > 0.8 together ⇒ write it down:
*seat-swap is the variance floor for greedy self-play A/B; more Rounds is the
only lever*. Stop proposing variance-reduction estimators for the harness.
Otherwise build the passing lever into `collect_pair_deltas`.

## 3. Build list (test-first)

1. `src/tichu_eval/aivat_probe.py`: critic loader (from `_rollout_weights.pt`
   via `save_rollout_weights`' layout), checkpoint recorder on
   `decision_observer`, GT-split MC completion sampler (keeps each seat's 8,
   reshuffles the other 24).
2. Tests:
   - completions preserve the 8-card prefixes and the 56-card multiset.
   - E-term mean → 0 on a synthetic V.
   - an identical-agents pair gives d ≡ 0 and d̃ ≡ 0.
   - team-A sign convention survives the seat swap.
3. `src/tichu_training/cli/aivat_precheck.py`: sharded and resumable like the
   pMCPA/piKL A/B runners; writes `deltas.npz` + a per-Position parquet.
4. Offline analysis script for M1–M4 (bootstrap CIs on every ratio, paired by
   Position).

**Cost:** about one drift-benchmark pair run, plus critic forwards (~10 per half),
plus K = 16 critic forwards on GT-divergent Positions only. Launch detached per
the usual training-launch rule.

**Build-time checks:**
- the critic's output units and seat convention.
- whether it was trained on pre-play (call/Schupfen) states. If it is
  out-of-distribution there, M1 falls back to the first Play state. That
  state includes Schupfen, so label M1 as "deal+exchange" luck.

## 4. Where AIVAT is actually the right tool (follow-up, not in scope)

In **unpaired** data the deal term does not cancel, and nothing else removes
it. That covers the served agent vs humans (live tapes) and human-vs-human
games (BSW corpus). There, `score − [V(deal) − E V]` is the DeepStack-style
estimator. Summed per player, it gives the luck/skill split for human players:
variance of player means after vs before the correction. M1 says how much it
would buy.
