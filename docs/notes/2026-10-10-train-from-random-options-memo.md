# 2026-10-10 — Options memo: train a Tichu agent from random (no BC)?

## Status (2026-10-12): CLOSED. Training from random is dead

The Shrink-and-Perturb Probe ran (§8, §9). Even a calibrated 5-point dent in the v7
champion is never won back: training against the real champion slid it from −5.3 to
−11.0 (CI [−12.1, −10.1]) over 4096 iterations. RL here cannot rebuild lost strength,
let alone build it from zero. The plateau is set by the **learning signal**, not by
the BC starting point or the KL leash. The leash limits damage, not gains (§9 corrects
§2's "slack" reading). Arm C was skipped on purpose. **Do not re-propose a
from-random or reset-based co-train without first fixing the signal.**

## TL;DR (original, 2026-10-10)

**Don't build a from-random (tabula rasa) co-train.** It only beats the current
approach if the v7 plateau comes from **where training started** (a basin around the
BC, or a net that lost plasticity) rather than from the **learning signal**. The
existing evidence points to the signal. Even in the favourable case, a from-random
run costs months of this box and gives up the human-partner compatibility that BC
provides, and the seat-swap tournament can't detect that loss.

The underlying hunch ("the BC is holding us back") is testable for about two days of
compute with a **Shrink-and-Perturb Probe**: blend the v7 champion's play net toward a
fresh init, re-train it with the v7 recipe, and see whether it climbs past the
champion or settles back to the plateau. The probe is pre-registered below and built
(`scripts/shrink_perturb_champion.py`, `scripts/calibrate_shrink_perturb.py`,
`configs/cotrain_v7_snp_{a090,a100}.yaml`).
Warm-starts are staged under `data/snp_v7/`. **Arm P v1 ran and was confounded: see §8.**

## 1. Framing

- **Objective.** A stronger served agent than the v7 champion (promoted at iter
  34304), i.e. one that gets past the ~+15-over-BC plateau.
- **Success criterion.** A CI-clean win over the v7 champion in the paired seat-swap
  tournament, while staying compatible with human partners (cpfix3328 is served
  alongside humans).
- **Constraints.** One machine, about 64 rounds/s (512 rounds per ~7.8 s iteration),
  roughly 5.6M rounds/day. No concurrent heavy jobs (ADR-0034).
- **Non-goal.** A different play *style* for its own sake. If that becomes the goal,
  the ranking below changes.

## 2. Evidence, read from `data/runs/cotrain_v7_gated`

| Observation | Numbers | Reading |
|---|---|---|
| Plateau | gate margin vs v7 BC: 12.0 @4096 → 14.3 @12288 → 14.6 @24576 → 15.8 @36864 → 14.3 @40704. 8 promotions, each about +1 over the previous champion, the last at 34304. Champion vs cpfix3328: +7.4 | About 28k iterations (~15M rounds) bought about 1 point |
| Play leash is slack | `play_kl_coef` between 1e-4 and 0.0256 for all 40k iterations; `play_kl` ≈ 0.10–0.17; `reanchor_on_promote` moves the anchor at each promotion | The BC anchor is not what caps **play**. *(Corrected in §9: play_kl hovered at its 0.12 target, so the leash WAS active, holding the policy against drift; it limits damage, not gains)* |
| Flaws in the starting point survive training | −5.35 call deficit inherited from the BC (−10.58); wishfix's −2.69 traced to a pre-fix schupfen warm-start (ADR-0040) | Final ≈ start + a bounded local gain. Caveat: calls and schupfen *are* tightly leashed (targets 0.008–0.02) |
| Prior diagnoses | advantage-SNR probe (critic epistemic bias), piKL variance wall (ADR-0037), gate CI false nulls | Every diagnosis so far blamed the signal, not the starting point |

## 3. Options

| Option | Mechanism | Load-bearing assumption | Cheapest falsifier | Verdict |
|---|---|---|---|---|
| **A. Continue BC-anchored co-train** (baseline) | Already running, gated, ratcheting | — | — | Plateaued (above) |
| **B. From random**: DouZero-style DMC/PPO self-play, no BC | Escapes the human basin and finds strategies humans don't play (the AlphaZero story) | (1) A better strategy exists that you can't reach from BC, **and** (2) this model-free signal can climb from zero within budget | 1-day random-init smoke run measuring rounds-to-beat-RuleAgent | **Not recommended; deferred behind the probe** |
| **C. BC start, anchor removed** (decay KL to 0, Ataraxos-style) | Removes the pull back towards human play | The leash is what limits | KL-coefficient trajectory | **Effectively already true for play.** Calls are a closed lever (2026-07-29). *(§9: removing the leash makes things worse, not better)* |
| **D. Shrink-and-Perturb Probe** | Restores plasticity / leaves the basin while keeping most of what the net knows | The plateau is caused by the starting point (location or lost plasticity), not by the signal | This probe | **Run it.** Tests B's assumption (1) for about 1/20th of B's cost |

### Why B fails under adversarial reading

1. **Its two assumptions pull against each other.** If the signal is strong enough
   to climb from zero to above +15, it should also be strong enough to climb out of
   the BC region once the leash is off, and for play the leash already is off. B
   beats C/D only if there is a genuine basin trap.
2. **Compute.** DouZero's reference point is about 1e10 environment steps on
   Doudizhu, a smaller game (3 players, no partner, no schupfen, no calls). At about
   5.6M rounds/day that is on the order of months, just to reach human level. The
   whole v7 run was about 21M rounds, starting from 756M rows of human play. The
   perfect-info critic (ADR-0033) is the one aid that suits from-scratch training (its
   gains are sample efficiency, per the 2026-07-26 sweep), and it already exists.
3. **Partner conventions.** Pure self-play in partnership games converges on private
   conventions (Hanabi: self-play agents collapse with human-like partners). The
   seat-swap tournament seats each agent with a copy of itself, so it **cannot detect
   this** (2026-07-26 sweep, §6). B could win our eval and lose in production.
4. **Literature.** The claim that a self-anchor (EMAgnet) can replace the BC anchor in
   Tichu was refuted 1-2 (2026-09-23 sweep). ADR-0033 held DMC-from-scratch only as
   the fallback if warm-started PTIE-PPO stalled. This memo explains why the
   fallback isn't triggered yet.

## 4. The deciding question

**Is the plateau caused by the starting point, or by the learning signal?**

From-random only helps in the first case. In the second it is strictly worse: the
same weak signal has a much longer climb. D answers this directly.

## 5. Pre-registration: Shrink-and-Perturb Probe

**Arms** (run sequentially; identical except for the warm-start):

| Arm | Config | Warm-start | Purpose |
|---|---|---|---|
| **P** (v2) | `configs/cotrain_v7_snp_a090.yaml` | v7 champion, play net blended `p ← 0.9·p + 0.1·p_fresh` (seed 0; the calibrated pick, iteration-0 margin −5.3); schupfen/tichu/grand copied unchanged; `_champion.pt` seeded with the unperturbed v7 champion. Parameter-space change: rel 0.099, norm 720.5 → 650.0. *(v1 used `a060`, α=0.6, unseeded champion: confounded, see §8)* | The treatment |
| **C** | `configs/cotrain_v7_snp_a100.yaml` | v7 champion, unchanged (α=1, β=0) | Controls for everything a fresh run resets: Adam state, critic (re-warmed), KL controllers, gate pool |

**Shared recipe.** The v7 recipe verbatim (`model:` block onwards) except:

- Play and wish KL leash off (coef pinned at the 1e-4 floor). In arm P the anchor
  would be the perturbed net, so a leash would trap the learner and the probe would
  measure the leash. The v7 run's play coefficient sat at 1e-4..0.0256, so this
  matches it in practice.
- A 6144-iteration budget (48 gate windows, ≈13 h of rollout/update per arm, plus
  gate time).
- Gate opponents: the arm's own champion (initially = its warm-start), the v7 run's
  `_bc_opponent.pt` (copied in, so the vs-BC stream is comparable window-for-window),
  and **v7champ** observe-only (`require:false`, `rollout:false`).

**Primary readout.** After the budget, run a ship-check tournament (template
`scripts/ship_check_wishfix_vs_cpfix3328.py`: greedy, seat-swapped, paired cluster
bootstrap, 40k deals) of each arm's final `_champion.pt` against the v7 champion.
Δ_P and Δ_C are the paired mean margins. If arm C never promotes, its champion *is*
the v7 champion and Δ_C = 0 by construction.

Why not read the gate stream: promotions reset every gate pool, including
observe-only ones (`PromotionGate.conclude`), so the v7champ stream's n depends on
promotion timing. It is the secondary (trajectory) readout only.

**Manipulation check.** In P's first gate window (iter 128), the v7champ margin must
be at least 3 points below C's. *(Superseded for v2 by the [−15, −3] band measured
before launch, §8.)* Otherwise α=0.6 was too gentle: rerun P at
α=0.4/β=0.6 before reading anything.

**Decision rule:**

| Outcome | Condition | Action |
|---|---|---|
| **Start-limited** | Δ_P CI_lo > 0 **and** Δ_P > Δ_C | The plateau depends on the starting point. Adopt periodic shrink-and-perturb in the co-train (cheap), and only then scope a from-random run, behind a cross-play / human-proxy eval |
| **Restart-limited** | Δ_C CI_lo > 0 and Δ_P ≤ Δ_C | A fresh optimizer/critic restart breaks the plateau; perturbation adds nothing. Investigate the restart (Adam state / critic staleness), not tabula rasa |
| **Signal-limited** | Δ_P CI_hi < +1.0 and Δ_C CI_hi < +1.0 | **Close from-random training for good.** Record in an ADR/memory: the plateau is set by the signal/objective. Effort goes to the signal (advantage-SNR levers) |
| **Inconclusive** | anything else | If P's gate stream is still rising over its last 10 windows (by more than 1 point across them), extend P once to 12288 and re-read; otherwise report as inconclusive |

## 6. How to run

From the worktree root, with this worktree's `src` first on `PYTHONPATH`. Both
arms were staged on 2026-10-11 (warm-starts in `data/snp_v7/{a090,a100}`, run
dirs seeded with `_bc_opponent.pt` and `_champion.pt`; P v2's play net verified
bit-identical to the calibrated blend). Re-stage only if the source champion
changes; the manifest records its sha256.

```powershell
python scripts/shrink_perturb_champion.py --weights C:\workbench\tichu\data\runs\cotrain_v7_gated\_champion.pt --config configs\cotrain_v7_snp_a090.yaml --alpha 0.9 --beta 0.1 --seed 0 --out-dir C:\workbench\tichu\data\snp_v7\a090 --run-dir C:\workbench\tichu\data\runs\cotrain_v7_snp_a090 --bc-opponent C:\workbench\tichu\data\runs\cotrain_v7_gated\_bc_opponent.pt --seed-champion
python -m tichu_training.cli.train_cotrain --config configs\cotrain_v7_snp_a090.yaml
```

Then the same for `a100` (α=1.0, β=0.0, also `--seed-champion`), after P finishes.
Launch detached (`Start-Process` with the console redirected), not as a harness
background task.

## 7. What would change the recommendation

- The probe returns **start-limited**: from-random becomes worth scoping, but only
  with a cross-play / human-proxy eval in place first.
- The served bot **never sits with human partners**: problem 3 goes away. Problems 1
  and 2 remain.
- A **search improvement operator** that survives the τ-continuation bias appears:
  AlphaZero-style from-scratch learning would then have a strong signal. Today it
  doesn't (ADR-0036/0037).

## 8. Addendum 2026-10-11: arm P v1 was confounded; calibrate the blend first

**What happened.** Arm P ran from 2026-10-10 14:39 and was stopped at iter ~4690
(of 6144). Per-window margins, recovered from the moving-pool gate means:

| Opponent | iter 128 | ~900 | ~3000 | 4608 |
|---|---|---|---|---|
| v7 champion | −58 | −30 | −32 | −38 |
| v7 BC | −41 | −15 | −15 | −22 |
| its own starting weights | +87 | +108 | +110 | +105 |

It recovered fast for about 900 iterations, stalled about 30 below the champion and
15 below BC, then slipped. No promotion ever fired.

**Why it is not a readout.** Two design flaws, both in the setup, not the idea:

1. **The blend wrecked the policy instead of perturbing it.** The α=0.6/β=0.4
   blend scores **−120.6** vs the champion at iteration 0 (64-deal smoke, CI
   [−148, −93]). That is closer to a partial from-random start than to "keep most of
   what the net knows". §5's manipulation check had only a lower bound (at least 3
   below C); it needs an upper bound too.
2. **A frozen broken opponent.** `also_beat_bc` blocks promotion until the learner
   beats BC, which it never did. So the rollout opponent stayed the wrecked
   starting weights for half of all games, all run. Those are easy wins with little
   signal, consistent with the slide after iteration 3000. Arm C would train against
   the real champion, so P and C had different opponents.

The weak signal it does carry: from a damaged start, training only recovered
partway and plateaued below BC. That fits "RL refines locally", but flaw 2
confounds it, so it does not close the question.

**Fixes for P v2:**

- **Calibrate α before training.** `scripts/calibrate_shrink_perturb.py` blends
  the champion at each α against the same fresh init (seed 0) and plays it against
  the unperturbed champion. The tournament is greedy and seat-swapped, with a paired
  bootstrap, on the same 4096 deals for every α (deal seed 190,000,000). It picks the
  **smallest α whose mean margin lies in [−15, −3]**. α=1.0 runs as a harness
  check: champion vs itself must read exactly 0 (smoke: +0.00). Output:
  `data/snp_v7/calibration/calibration.csv` and per-α deltas.
- **Seed P's `_champion.pt` with the v7 champion**, so the rollout opponents are
  {v7 champion, BC}, identical to arm C. The arm's "champion" gate stream then
  measures P against the v7 champion directly.
- **Manipulation check becomes a band:** P's iteration-0 deficit must lie in
  [−15, −3]. The calibration measures exactly that, so the check is now satisfied
  before launch rather than read off the first window.

**Calibration result (2026-10-11).** 4096 deals per α, seat-swapped, paired 95% CI,
blend vs the unperturbed v7 champion at iteration 0:

| α (β = 1−α) | param rel change | margin | 95% CI | |
|---|---|---|---|---|
| 1.00 | 0.000 | +0.00 | [+0.00, +0.00] | harness check passed |
| 0.95 | 0.050 | −1.22 | [−3.31, +0.88] | too gentle |
| **0.90** | 0.099 | **−5.31** | [−7.80, −2.86] | **pick** |
| 0.85 | 0.149 | −15.23 | [−17.96, −12.42] | just outside the band |
| 0.80 | 0.199 | −34.01 | [−37.19, −30.85] | too strong |
| 0.70 | 0.298 | −78.35 | [−81.74, −74.96] | too strong |
| 0.60 | 0.397 | −120.6 | [−148, −93] (64-deal smoke) | P v1's blend |

The deficit is steeply non-linear in β: it roughly doubles every 0.05 of β past
0.10. **P v2 uses α=0.90, β=0.10**, the pre-registered pick (the smallest α whose
mean is in band). Its start is at the gentle end of the band. A mid-band α (~0.875)
was not chosen after seeing the data, to keep the pick rule honest.

## 9. Addendum 2026-10-12: arm P v2 result, arm C skipped, verdict

**P v2** (`configs/cotrain_v7_snp_a090.yaml`: α=0.9, champion seeded with the v7
champion, leash off) ran from 2026-10-11 17:27. The owner stopped it at iteration
4223 of 6144. The pre-registered extension rule did not apply because the stream was
falling, not rising. It was never promoted. Learner vs the v7 champion, per
128-iteration window, recovered by differencing the gate's moving-pool means:

| Iterations | 0 (calibration) | 128–512 | ~1000–2000 | ~2000–3000 | last 4 windows (to 4096) |
|---|---|---|---|---|---|
| vs v7 champion | −5.3 | −6.4, −5.3, −4.4, −6.1 | −7 to −11 | −9 to −13 | −8.7, −12.7, −13.6, −12.2 |

- **Pooled at iteration 4096** (last 10 windows, 81,920 paired observations): **−11.04,
  CI [−12.06, −10.07]** vs the v7 champion; +5.65 [+4.51, +6.77] vs BC (it started
  near +9).
- **Play KL from the starting weights** grew 0.21 → 0.43 with no gain.

The run trained against the real champion the whole time: the `champion` and
`v7champ` streams are identical (the seeded `_champion.pt` is byte-identical to the v7
champion, and it fills half of all rollout games).

**Readout note.** §5's primary readout (a 40k ship-check of each arm's *final
champion*) is degenerate once the champion is seeded. An arm that is never promoted
still holds the v7 champion, which would read exactly 0. The learner's gate stream
(paired, 81,920 observations) is the readout used here.

**The leash, corrected.** §2 read the v7 run's small `play_kl_coef` as "leash slack".
That was wrong. `play_kl` hovered at its 0.12 target, so the controller was holding
it there. The leash was active, and what it held back was **drift, not gains**. After
the 34304 promotion the anchor sat on the champion, and the leashed learner was
CI-clean *below* that champion for 6,400 iterations (−1.9, −2.6, −1.5, −2.4, −1.3,
−1.3 per 1024 iterations). P v2, unleashed, drifted further (KL 0.43) and lost more.
Free movement on this signal is a downhill random walk. The gate's ratchet keeps the
rare validated step (about +1 per promotion), which is why progress is slow.

**Arm C skipped (owner decision, agreed).** C (undamaged champion, leash off) would
cost about 14 hours to confirm what the late v7 run already shows: inside the leash
the learner was already losing ground, and P v2 unleashed lost more. If the leash
question ever needs nailing down, a short C (about 1000 iterations) shows the
direction in half a day.

**Verdict.**

- **Training from random is closed.** Its load-bearing assumption (2), "this signal
  can climb from zero", is refuted: the signal cannot even win back a 5-point dent
  with the champion as its opponent. Assumption (1), the basin, is moot.
- **Reset-style levers are closed too** (shrink-and-perturb, partial re-init,
  plasticity resets). The net does not need its learning ability restored; the
  signal has nothing reliable to teach it.
- **The plateau is signal-limited.** The remaining levers are signal quality: the
  pre-registered advantage-SNR levers (critic epistemic bias; stratum-weighted value
  loss) and paired- or duplicate-deal training signals (the blunder-miner line).

**Reusable from this probe:** `tichu_training.shrink_perturb`,
`scripts/shrink_perturb_champion.py` (including `--seed-champion`), and
`scripts/calibrate_shrink_perturb.py`. The calibration is a quick way to measure how
much a weight-space change costs in play before training on it. `a060` (v1) and
`a090` (v2) are kept as the record; `a100` (C) is staged but never run.
