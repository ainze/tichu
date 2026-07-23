# 2026-07-03 — Trunk-capacity probe at featurizer v6: pre-registration

**Written before any probe number is seen.** Design, bar, and closing rule are
committed here first (the ADR-0039 lesson: a capacity/residual verdict read
after the fact, without a pre-registered bar and early-stopping discipline, is
untrustworthy).

## What reopens a closed question

[2026-06-02-bc-is-label-bound-not-capacity-bound.md](2026-06-02-bc-is-label-bound-not-capacity-bound.md)
closed trunk capacity: a 2.25× trunk (1536×6) tied baseline (+0.08 pp play-acc)
→ "trunk size is settled: do not touch it." The
[bc_full_corpus_v6_memmap.yaml](../../configs/bc_full_corpus_v6_memmap.yaml)
header cites it as the reason arch stays 1024×4.

This probe is **not** a re-roll of that experiment. The reopening basis is that
note's own caveat #2:

1. **The 06-02 sweep ran on v4 features (224 dims).** Featurizer v6 (ADR-0038)
   grew the input to 591 dims of *new content* — per-player `played_by`,
   `schupfen_received`, cross-Trick negative-info, `trick_leader`. A trunk
   sized when the input carried less signal can be the binding constraint now
   that it carries more. "Input got wider" is NOT the claim (input width only
   changes the first Linear, +0.4M params); "the learnable ceiling may have
   risen" is.
2. **The 06-02 metric was top-1 accuracy**, pinned near the ~85% Bayes ceiling
   of human disagreement and nearly blind to fit gains. This probe gates on
   NLL, which keeps moving after accuracy saturates.

Hypothesis under test: **the play head's BC fit at v6 is capacity-limited at
1024×4.** Nothing else — no cotrain-strength claim, no critic claim.

## Arms (each = baseline config + ONE arch delta)

Baseline: the **finished** `bc_full_corpus_v6_wishfix_memmap` run (1024×4,
out 512, head 256) — reused, not retrained. All arms share its bundle
(`materialised_full_v6_wishfix\bc` — play rows byte-identical to plain v6),
LR 3e-4, batch 1024, epochs 1, head_weights, shuffle config, **seed 0**.
Same seed + shuffle ⇒ identical example order ⇒ **paired** comparison.

| arm | config | delta | tests |
|---|---|---|---|
| depth  | [trunk_1024x8](../../configs/bc_full_corpus_v6_wishfix_trunk_1024x8_memmap.yaml) | depth 4→8 | depth axis (head-probe prior: depth was the lever for tichu/schupfen) |
| width  | [trunk_2048x4](../../configs/bc_full_corpus_v6_wishfix_trunk_2048x4_memmap.yaml) | hidden 1024→2048 | width axis; ~param-matched to depth arm (~+8-9M) so shape ≠ count |
| funnel | [trunk_funnel](../../configs/bc_full_corpus_v6_wishfix_trunk_funnel_memmap.yaml) | out 512→1024, head 256→512 | the 512→256→1809 funnel after the trunk; near-zero inference tax if it wins |
| both   | [trunk_2048x8](../../configs/bc_full_corpus_v6_wishfix_trunk_2048x8_memmap.yaml) | hidden 1024→2048 AND depth 4→8 | ~4× trunk params — bounds what any single axis could reach; doubles as the cotrain negative-control's warm-start |

Arms train **from scratch** (shape mismatch forbids warm-start) and run
**sequentially** (ADR-0034: no concurrent heavy jobs). LR/schedule held fixed
across arms — declared caveat: a larger trunk might prefer a different LR; a
null here is a null *at the production recipe*, which is the deployable
question.

## Metric

- **Primary:** `loss_play` (masked, sample-weighted cross-entropy = NLL) from
  `step.csv`, averaged over the **trailing 50,000 play batches** (~51M
  examples). At epochs=1 every batch is unseen data, so trailing train-stream
  NLL is honest held-out NLL (progressive validation); the shared shuffle
  order makes it paired.
- **Secondary (reported, never gated):** Move Prediction top-1/top-5 on the
  Held-out Game Set (`eval_matrix --mode move_prediction`), decile-9 slice if
  cheap. No critic-R² anywhere (ADR-0033/0039).

## Pre-committed bar

Adopt an arm only if **all** of:

1. Trailing-50k `loss_play` beats baseline by **≥ 1% relative**.
2. The gap **holds across the trailing curve** (per-10k-step buckets over the
   final 200k steps all favour the arm — not an endpoint artifact). The paired
   design replaces a seed-noise σ requirement.
3. Move Prediction top-1/top-5 does not regress.

Tie-break if several arms clear: lowest inference FLOPs wins
(funnel < depth ≈ width).

**Adoption meaning:** the winner becomes the production wishfix BC and the
warm-start for the gated cotrain (`model:` block updated everywhere,
export/serve model-configs regenerated to match the new shape). It buys a
cotrain *experiment* with the bigger trunk — not a strength claim; the
leash-anchors-to-BC argument (head-capacity probe, 2026-06-19) is why a better
prior is expected to propagate.

## Negative control: 2048×8 cotrain vs the shipped champion

Separate from (and regardless of) the BC bar, the 2048×8 BC seeds a cotrain
falsification run
([cotrain_v6_pbrs_resid_wish_gated_wishfix_trunk2048x8.yaml](../../configs/cotrain_v6_pbrs_resid_wish_gated_wishfix_trunk2048x8.yaml)):
byte-identical recipe to the 1024×4 wishfix gated run except the play trunk.
Purpose: test the residual worry that BC-fit is the wrong lens and trunk
capacity only pays off *under cotrain* (the leash anchors to BC, but PPO can
in principle use capacity BC can't).

- **Instrument:** out-of-loop `check_cotrain` vs `eval.master` (the shipped
  cpfix iter-3328 champion) on the 2048×8 run's snapshots, at promotion iters
  and round-thousand marks. Retroactive matched pairs from the 1024×4 run are
  NOT available (its intermediate snapshots were deleted, 2026-07-22); the
  comparison basis is its recorded early band (−15.9 @ 128, −10.4 @ 256,
  −9.9 @ 512) and its known endpoint (still below master at ~18.5k iters).
  Null = the 2048×8 trajectory plateaus below master likewise, never clearly
  ahead of that band at comparable depth. Compare **per-iteration**, never
  per-wall-clock-hour (the big trunk's iterations are ~4× more expensive —
  that cost asymmetry is itself part of the verdict).
- **Running record** (2048×8 vs master, greedy seat-swapped, n=8000):
  iter 1408 → **−11.66 [−16.61, −6.65]** — in the same band as the 1024×4 run's
  early trajectory (−15.9 @ 128, −10.4 @ 256, −9.9 @ 512); no capacity signal
  so far.
- **"Does not improve" (the expected null):** at matched iteration counts the
  2048×8 trajectory vs master is never CI-above the 1024×4 run's trajectory
  vs master. If that holds through the point where the 1024×4 run had clearly
  plateaued (~a few thousand iters), stop the run and record the null.
- **Surprise clause:** if it IS CI-above at matched iterations, that is a
  major finding (capacity binds under RL, not imitation) and gets its own ADR
  before any adoption — a bigger trunk taxes every rollout, gate, and serve
  forever.

## Closing rule (permanent)

If no arm clears: the trunk-capacity question is closed **permanently** —
tested twice, at both featurizer generations (v4/100k/accuracy 2026-06-02;
v6/full-corpus/paired-NLL here). The v6 config headers gain the second
citation. No third reopening on a future featurizer bump without qualitatively
new evidence (e.g. a demonstrated train-fit gap, which neither test found).

## Analysis

`python scripts\analyze_trunk_probe.py` — reads each run's `step.csv`,
prints trailing-window means, relative deltas, and per-bucket paired gaps.

## Result

_To be filled after the runs. Numbers must not influence the section above._
