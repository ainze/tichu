# 2026-06-02 — BC play head is label-bound, not capacity-bound

## TL;DR

A **2.25× larger trunk moved play accuracy by +0.08 pp** (noise). The ~84–85%
BC play-head plateau is a **label / data ceiling**, not a representational
capacity limit. This closes followup #3 ("bigger BC architecture") from the
[AWR game-target note](2026-05-29-awr-game-target-flat.md): a wider trunk will
not unlock play-head strength. The lever for a *stronger* player is online RL
(sharpen toward winning, not toward matching the human top-1 move) — not a
bigger imitation model.

## The experiment (already on disk)

The capacity sweep was run at v4 / 100k, arch held constant except size
(see [configs/bc_xl_100k_v4_memmap.yaml](../../configs/bc_xl_100k_v4_memmap.yaml),
whose header states the gating rule verbatim: *"bc_xl ties or loses → data-bound;
skip capacity"*).

| Run | Trunk | Heads | Params | Final play-acc¹ |
|---|---|---|---|---|
| `bc_full_100k_v4_memmap` | 1024 × 4 | 256 | 1× | **0.8395** |
| `bc_xl_100k_v4_memmap`   | 1536 × 6 | 384 | ~2.25× | **0.8403** |

¹ mean `acc_play` over the last 3000 play-head batches of `step.csv`.

Δ = **+0.0008**. It tied. Per the config's own gating rule: **data-bound.**

## Why this is a label ceiling, not capacity

1. **Not capacity** — 2.25× more params did not fit the *training* data any
   better. A capacity-bound model improves when you add capacity; this didn't.
2. **Not overfitting** — train-acc (~84–85%) ≈ held-out (~84%, per the AWR
   note). No train/val gap.
3. **Bayes ceiling of imitation** — the same `PrivateState` maps to different
   human moves in the **BSW Corpus** (two strong players legitimately make
   different legal plays in the same position). When the target is intrinsically
   stochastic, ~85% top-1 agreement is irreducible. Fast convergence fits this:
   the engineered v5 featurizer surfaces the learnable signal almost
   immediately; the residual 15% is human disagreement, not unlearned structure.
4. **Data scale agrees** — 100k → full corpus (~100× data) moved play-acc only
   ~84% → ~85%. Not data-*volume*-bound either.

## Caveats

- The sweep was **+50%/+50% (1536×6)**, not the 2048×6 the AWR note floated, and
  it was **v4 / 100k**, not v5 / full corpus. A far larger trunk *might* eke out
  another fraction — but +2.25× params → +0.08 pp is a textbook
  diminishing-returns verdict. A 2048×6 BC is very unlikely to pay for itself.
- "The trunk can't imitate better" (proven here) is **not** the same as "the
  512-dim representation carries everything a strong player needs." The latter is
  bounded by the **Featurizer** *content* (e.g. it encodes no opponent-hand
  belief), not the trunk *width*. That is the **Belief Model** lever, orthogonal
  to both trunk size and PPO.

## Bearing on the plan

Confirms the bottleneck is not representational capacity — imitation is at its
intrinsic ceiling. Strengthens the case for **PPO Refine** (online self-play
sharpening) over any further BC scaling. Trunk size is settled: **do not touch
it.** See [CONTEXT.md](../../CONTEXT.md) §"Phase 2 / online-RL terms".
