---
status: proposed
---

# Frozen-reference vine + pooled-verdict ratchet (vine v3)

The 2026-07-24/25 diagnose sessions
([2026-07-24-advantage-snr-probe-critic-bias.md](../notes/2026-07-24-advantage-snr-probe-critic-bias.md))
established that (1) the co-train plateau is caused by advantage corruption in
call-hinge/late-round states — the Perfect-Info Critic's residual there is
50–77% fixable-in-principle error, but every state-function fix is exhausted
(capacity, type, perfect-info inputs, loss reallocation, and Claim-Solver
certainty inputs all null — the missing information is *policy-deterministic
outcomes*, i.e. a playout); (2) exploration is dead (sampling ≈ argmax); and
(3) the recoverable pool persists on the current champion (0.32 robust
blunders/round, ~+15–34/round). The one estimator that carries playout
information with zero variance is the vine (ADR-0035) — closed after
fixed-anchor flatness (ball-capped) and a −23.59 re-anchoring death
(self-play co-drift). This ADR funds its named-but-unbuilt successor with the
two failure modes fixed by components built since: a **frozen Reference
Field** for continuations, and the **greedy gate** (proven-faithful deployed-
strength signal) extended with a **Pooled Verdict** so provable-but-small
steps can bank.

## Decisions (grilled 2026-07-25, each locked with rationale)

1. **Reference Field = the rolling gate champion.** All four seats of every
   branch playout are played by the champion's weights, frozen between
   promotions. On a gate promotion, champion, play KL-anchor, and Reference
   Field advance **atomically** to the same weights. **Invariant: the
   Reference Field may only ever be set to weights that have passed the
   greedy gate — never advanced on a timer, never the live learner.** This
   single invariant is the difference from the −23.59 escalation. (Rejected:
   cpfix3328-forever — goes stale-inconsistent as the learner passes it, the
   piKL τ-continuation family; a lagged reference — evaluator/opponent
   inconsistency for no proven benefit.) The estimator is then conservative
   policy iteration: each vine advantage estimates Q_ref(s,a) − mean-branch
   Q_ref; the learner improves against a fixed evaluator and the evaluator
   advances only on CI-proven wins.
2. **Row sourcing.** Trunk games: learner argmax (gradient lands on states the
   learner actually reaches). Chosen branch is **replayed under the Reference
   Field** (the v2 parity trick is invalid across policies; cross-policy
   returns would reintroduce the bias family this design kills). Decision
   selection is **stratified toward the pool**: prefer states with hand ≤ 10
   or a live caller, uniform fallback — v1's near-tie dilution burned 3/4 of
   its budget on ~zero-advantage rows.
3. **Estimator dials = v2's proven settings.** 64 games × 4 decisions/iter,
   top-3 learner-ranked alternatives (+ chosen replay = 4 reference playouts
   per decision, ~1,090 playout-equivalents/iter, ~+5 s/iter), `emit_branches`
   with **positive-advantage alternatives only** (the v2 NaN fix; the chosen
   row keeps both signs), `min_abs_advantage: 5`, per-net normalization,
   `old_logp` = policy's logprob of the forced action. Vine rows **replace**
   the play-GAE group. Every branch is played **fully to round end** — the
   return is the complete round outcome (points + call bonuses + slam),
   which is what makes rows exact given the world. Vine deal stream seed
   900,000,000 (v1's 555000 collides with rollout seeds near iter ~1084).
4. **Ratchet arithmetic — the Pooled Verdict.** The per-window gate needs a
   true +~5 edge to promote at n=8,192, but a KL-ball step of vine
   improvement is realistically +0.5–2: the old gate would reproduce the
   stall by arithmetic even in the success world (the λ=1 arm demonstrated
   exactly this signature). Fix: gate margins **accumulate across consecutive
   windows against the unchanged champion** (fresh deals per window make
   pooling valid); the verdict each window uses the pooled sample; promote on
   pooled CI_lo > 0; the pool resets on promotion. A true +1.5 edge banks in
   ~10–12 windows; winner's-curse exposure drops (no more single-lucky-window
   promotions). Caveat on record: the pooled margin measures the drifting
   learner's trajectory average while promotion elevates current weights —
   conservative under a rising trend, mildly optimistic under oscillation,
   bounded by the pooled CI.
5. **Leash geometry.** Play KL target **0.02** (the best-track-record leash;
   the wishfix 0.12 was a plateau-escape under a *biased* gradient) with
   `reanchor_on_promote` — total travel is the sum of gate-proven steps.
6. **Everything non-play is frozen recipe.** Main-rollout λ stays **1.0**
   (the remaining GAE consumers are the earliest-state one-shot decisions,
   maximally exposed to the proven-unfixable critic bias at λ<1; unbiased-
   and-parked beats biased-and-nudged under their tight leashes). Head
   leashes/entropy byte-identical to the wishfix recipe; `cotrain_wish` on;
   the critic keeps training (targets raw R) and baselines the non-play
   heads only — the first run where play strength is fully decoupled from
   critic quality.
7. **Gate + launch mechanics.** Opponents {champion, bc} required +
   cpfix3328 observe-only (`require: false`); cadence 4096 deals / 128 iters
   / 10 workers. Warm start = the wishfix run's `_champion.pt` (last
   CI-proven weights; the unproven λ=1 drift is noise-grade and gets
   re-earned bankably if real), converted to `.bin` warm files. **Pre-seed
   the new run dir with the wishfix run's `_bc_opponent.pt`** — the trainer
   derives it from warm_start otherwise, which would silently turn the "bc"
   axis into a second champion axis. Fresh run dir; the λ=1 run stops at
   launch (verdict recorded).

## Pre-registered bars (committed before any number)

- **Valid dials, exhaustively:** pooled champion-axis margin + promotions;
  the cpfix3328 observe stream; out-of-loop `check_cotrain` vs cpfix3328 at
  promotions. **Invalid, on the record:** correction fix-rate (rose while
  strength collapsed), sampled rollout margins (entropy confound),
  value_loss.
- **Mechanism-works:** ≥1 pooled promotion whose post-promotion pooled margin
  stays ≥0 for the next 8 windows, with a non-degrading cpfix3328 stream.
- **Ship:** champion beats cpfix3328 seat-swapped n=40k, 95% CI > 0, AND
  Move-Prediction top-1 within −3pp of the warm-start champion (the eval
  double-bind / drift guard).
- **Stall/kill ladder:** stall = pooled champion-axis CI entirely below +1
  after 16 consecutive windows vs the same champion. First stall → the one
  pre-planned escalation (`games_per_iter` 64→128). Second stall → **kill
  permanently**: the paired-playout family is exhausted in this codebase
  (fixed-anchor flat; blind re-anchor −23.59; frozen-reference + pooled
  ratchet + enriched rows + stratified states stalled) and no vine variant
  returns without qualitatively new evidence.
- **Safety kill, every window:** pooled champion CI_hi < −2, or the
  cpfix3328 stream CI-clean below its −2.4 baseline, or bc-axis collapse →
  stop and autopsy; a clean gradient regressing means a bug, not tuning.
- **Hard stop** ~10k iterations; decision point ~40 windows.

## Consequences

- Play-head training no longer consumes the critic anywhere; critic quality
  questions are moot for play strength for the duration of this design.
- The gate gains a cumulative mode (`PromotionGate` extension next to the
  observe-only feature); all prior per-window semantics remain available.
- Cost ~13–14 s/iter (+ ~5 s vine) plus one extra observe-only
  mini-tournament per window — ~35–40 min/window all-in.
- If killed at the ladder's end, the strength program's remaining paths are
  outside the self-play-gradient family entirely (human-measurement loop /
  accept the plateau) — that closure statement is deliberate and part of the
  bar.

## Outcome (2026-07-26) — KILLED: permanent paired-playout family kill, owner-confirmed

The run (iters 0→1791, 13 gate windows, ~19 h) was **safety-stopped in
CI-clean regression**, not stalled: pooled champion-axis margin −2.19
[−3.48, −0.84] at n=98,304 — the learner losing to its own frozen warm-start
anchor — with zero promotions; the cpfix3328 observe stream (−3.73 [−5.05,
−2.36]) touched its −2.4 kill line at windows 1152 and 1280. The machinery
itself validated cleanly (pooled accumulation 8192→98k across held windows,
one verdict per window after the 0c8f38d latch fix, ~32 s/iter): the machine
worked; the gradient was the defect.

Three-round autopsy, no fixable bug found:

1. **KL-support hole refuted** (`scripts/diag_vine_v3_kl_support.py`): the
   leash is measured on the vine batch only (`update.py` kl_anchor_loss), but
   early states drift no more than pool states (KL 0.052 vs 0.041; argmax
   agreement 97.6% vs 94.4%). Root cause of the non-effect: `_is_pool_state`
   covers **94.6% of visited decisions** (callers live in most decile-9
   rounds) — §2's stratification premise was a near no-op in practice.
2. **Continuation flip refuted at sign level**
   (`scripts/diag_vine_v3_disagreement_replay.py`, 767 learner-vs-anchor
   disagreement states, both actions played out under field and self
   continuations): 84% sign agreement, flips symmetric (7.1%/8.8%). The harm
   itself is unresolvable at decision level — per-state paired-Δ sd ≈ 140,
   so confirming −2.2/round would need ~30k games (the ADR-0034/0035/piKL
   variance wall, met again).
3. **Chimera head-swap localization convicted the trunk**
   (`scripts/diag_vine_v3_chimera_gate.py`, three arms × 16,384 shared deals
   vs the champion; offline harness replicated the run's gate level):
   learner −2.32 [−4.71, +0.04]; **learner-trunk-only −2.53 [−4.80, −0.20]**;
   learner-standalones-only **+0.36 [−1.99, +2.75]**; paired contrasts
   additive (interaction −0.15). The deficit lives entirely in the trunk
   net — the vine-trained territory. (Wish rides the same trunk, but its
   0.008 leash and the λ=1 arm's 35 drift-free windows on the identical wish
   recipe make play the carrier.)

Verdict: the vine gradient produces a small, stable, trunk-local regression
via a mechanism below the resolution of every decision-level instrument
available. With v2 (self-continuation) flat and v3 (frozen-field) negative,
the family's best observed gradient value is zero — and a pooled ratchet
cannot bank zeros. Per the pre-registered bar: **no vine variant returns
without qualitatively new evidence.** As §Consequences committed, the
strength program's remaining paths lie outside the self-play-gradient
family (human-measurement loop) or in accepting the plateau.
