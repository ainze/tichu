# 2026-07-29 — Calls are not a strength lever; the gate was jammed by a statistics bug

## TL;DR

Session opened on "how do we improve exploration during cotrain — it feels stuck in a
local optimum." Exploration was never the constraint. Three findings, in order:

1. **The promotion gate could not resolve improvement.** It bootstrapped paired
   seat-swap deltas as unpaired, inflating its CI ~1.45x. It needed mean ≥ ~+4.8 to
   fire and so held **129 consecutive windows** while the learner won every one.
   Since promotion is the only thing that advances the KL anchor, that *was* the
   "inescapable local optimum". Fixed (`d47ee54`); first promotion in 129 windows
   fired immediately.
2. **Pooled verdicts could not forget a bad patch.** Unbounded pooling assumes a
   stationary learner. Added `pool_windows` (`28feee5`).
3. **The call-bonus deficit is inherited from the BC, and call heads do not matter
   at all.** This retires calls as a research direction.

## Calls: closed

The iter-27008 champion reads +0.137 total vs the served export from **+5.49 card
play / −5.35 call bonus**. Chased into the call subsystem for three experiments
(behavioral rates, success rates, threshold EV sweep) — all underdelivered. The
chimeras settled it: hold everything fixed, swap exactly one component.

### Call heads are interchangeable in every context tested

| heads compared | play policy | call-bonus delta |
|---|---|---|
| candidate vs served | candidate | +0.57 [−0.27, +1.37] |
| candidate vs served | served | −0.50 [−1.31, +0.28] |
| BC vs served | BC | +0.56 [−0.22, +1.38] |

BC-trained *and* cotrain-refined heads, on raw-BC *and* cotrain-refined play. All
three straddle zero.

**Why:** the call decision sits at a **flat EV optimum** — at the margin
P(success) ≈ 0.5, so marginal calls are EV-neutral. Corroborated three ways: the
threshold sweep recovered only +0.59 across a wide range; the behavioral profile
showed materially different call rates (0.189 vs 0.165 tichu) producing the same EV;
and three head transplants all returned ~0. That is a *well-calibrated* boundary,
not a broken one.

**Consequence:** "tailor the call network to the play policy" (the natural Stage 2
of a staged/decoupled training plan) has ~1 point of headroom, not ~10. Dead as a
strength lever. The deficit is **play conversion** — how well the policy turns a
call into going out first.

### Inherited, not generated

| pair | total | card play | call bonus |
|---|---|---|---|
| bc_wishfix vs served | −16.29 | −5.71 | **−10.58 [−12.42, −8.83]** |
| candidate vs served | +1.39 | +6.22 | −4.83 [−6.46, −3.24] |
| bc_wishfix vs candidate | −14.49 | −10.99 | **−3.50 [−5.12, −1.93]** |

Candidate-minus-BC: 27k iterations of co-training improved card play **+10.99** and
call bonus **+3.50**, both CI-clean. **There was no trade — both halves improved.**
The BC starts −10.58 behind and cotrain closed roughly half.

## The wrong turn, recorded

An earlier reading of this session concluded co-training *systematically trades
go-out-first competence for card points*. **Wrong in sign.** It came from a single
cross-lineage comparison (a 27k-iteration wishfix agent vs a 3.3k-iteration cpfix
agent, on different BCs) treated as a training effect. It looked corroborated when
the restart measured a champion-axis drift of −0.675 [−1.277, −0.073] over 3,200
iterations and −0.675 × (27000/3200) = −5.70 landed on the observed −5.35.

That was one borderline CI multiplied by 8.4. The direct measurement came back the
other way. **A level difference between lineages is not a training effect** — put the
starting point on the board before attributing a gap to what happened during a run.

## Gotcha: `_bc_opponent.pt` is not the BC

A run's `_bc_opponent.pt` is written lazily (`if not exists`) at gate setup, so under
`reanchor_on_promote` a resume after a promotion captures the **anchor** instead.
Measured on the wishfix run: its play net differs from the BC export by
**max |logit diff| 812**.

Two consequences: that run's gate `bc` stream (+9 to +15 throughout) was measured
against an early co-trained checkpoint, *not* human-imitation play — so
`also_beat_bc` was not the anti-cycling guard it was documented to be. And anyone
wanting "the BC" must export from the config's `warm_start` checkpoints:

```
py -m scripts.export_rollout_weights --warm-start \
    --config configs/cotrain_v6_pbrs_resid_wish_gated_wishfix.yaml \
    --out data/export/bc_wishfix_warmstart
```

That export is bit-identical (0.000e+00, head by head) to the standalone BC play
export; the `_bc_opponent.pt` one is not.

## Also found

- **PBRS was never implemented.** `ppo.pbrs {enabled: true, potential: banked_points}`
  is inert — nothing in `src/tichu_training` reads it. Every run in this lineage is
  *named* `..._pbrs_...` while running unshaped.
- **The call threshold is inference-only and cotrain cannot tune it.** Training
  *samples* from P(call); deployment *thresholds* at 0.5. A state at P=0.30 calls 30%
  of the time in training and never at deployment. Mostly moot given the flat-optimum
  finding above, but worth knowing.

## State at close

- **iter 27,008 is the best artifact**: +1.39 [−0.75, +3.58] vs the served export,
  positive across three independent measurements (+0.137, +0.90, +1.39), regression
  excluded, carrying the wish-decline fix the served export lacks. **Ship candidate.**
- ~~**The restart at a 0.02 leash is converged, not blocked.** `play_kl` sits at 0.0175
  *below* its 0.02 target — the policy is not using the movement it is allowed — and
  25 gate windows produced zero promotions. That is an in-ball optimum with a working
  ratchet, which is a different situation from the 129-window jam.~~
  **REFUTED by the session's final measurement — see below.**
- The open problem is back to play-policy improvement: better critic, different BC,
  or accept the plateau. Worth reopening as a fresh question rather than as a
  continuation of this run. **Qualified: a fourth path — delivering verified
  corrections as a pairwise ordering — was scoped afterwards in
  [ADR-0042](../adr/0042-preference-correction.md).**

## Correction: noise-limited, not converged

Stratified blunder mining (v1 protocol, both runs, both Control Bands clean at 0/150)
refuted the "converged" reading above. Per-**Delta Band** survival extrapolated back
to each band's tier-1 population — the only estimator that is comparable across runs:

```
                    [15,50)       [50,150)      [150,400)    control   rate/round  mean d   pool
cpfix3328 (served)   8/150  5.3%   5/150 3.3%    2/150 1.3%   0/150      0.324      105.7   34.3
iter 27008          15/150 10.0%   6/150 4.0%    5/150 3.3%   0/150      0.583       67.1   39.1
```

Co-training traded a few large **Verified Corrections** for many smaller ones and left
total identifiable loss ~unchanged. The policy has ~0.58 demonstrably-better actions
per Round against a margin over the served export of +1.39/Round. **Noise-limited,
not converged.**

Carry this qualification with the number: **~39 pts/Round is an upper bound, not an
achievable gain** — it sums individually-avoidable losses, and fixing one Decision
changes the others (the labels are policy-relative).

**A second correction, to the 2026-06-11 note.** Comparing this rate to that session's
headline "~0.2–0.3 blunders/Round" is invalid: that figure was a crude
`5.3 decisions/Round × 3–7%` blend, not the band-population extrapolation. Recomputed
on the same estimator, `iter_06225` is **0.374/Round** (pool 23.6). So the rate is
**1.56× higher, not 2.3×**, the per-Round candidate populations are identical to two
decimals (3.04/4.12/3.13 vs 3.01/4.22/3.41), and the survival difference is 17/450 vs
26/450 at **Fisher exact p = 0.211**. **The recoverable pool has not been shown to have
changed.** The 2026-06-11 kill note also under-stated its own ceiling: `0.374 × 0.175
× 63.1 = +4.1/Round`, not the "+2–3" recorded there.
