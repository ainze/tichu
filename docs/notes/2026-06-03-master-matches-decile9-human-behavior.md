# 2026-06-03 — Master ML Agent matches decile-9 human *behavior*; weakness is sub-behavioral

## TL;DR

The `master` tier (decile-9-conditioned BC, full-corpus v5) is **not behaviorally
soft**. Measured against decile-9 humans straight from the BSW Corpus, it bombs
*as often as the best humans* and calls Tichu/Grand only mildly less (an `argmax`
decoder artifact, near-free to fix). No gross behavioral pathology exists. The
human-beats-it weakness is therefore **sub-behavioral** — decision *quality*
within normal play, not a missing or mis-rated behavior. This kills the
"teach it to bomb" rationale for PPO and leaves the Bayes-ceiling sharpening
rationale as the sole survivor. Opponent-modeling (Belief/search) remains the
other live, untested suspect.

## How this came up

Owner can beat `master` in direct play and "hadn't seen it bomb." Built
**Behavioral Telemetry** (`eval_matrix --mode behavioral`, self-play) → master
bombs ~0.115/round (NOT zero; "never bombs" was a rare-event illusion). To judge
soft-vs-correct, built a human baseline by Skill Decile from the parquet manifest
([scripts/human_call_bomb_baseline.py](../../scripts/human_call_bomb_baseline.py)):
call-rate denominator = schupfen-shard rows (one per player-round); calls from the
call shards (positives only); bombs detected structurally from the play shard
(4-same-rank / 5+ same-suit-consecutive), sampled at 12.5% of row groups.

## Numbers

| | tichu_call | grand_call | bomb/round |
|---|---|---|---|
| human decile 0 | 0.102 | 0.016 | 0.105 |
| human decile 5 | 0.143 | 0.063 | 0.110 |
| human decile 8 | 0.145 | 0.110 | 0.109 |
| **human decile 9** | **0.145** | **0.132** | **0.110** |
| **master (ML d9)** | **0.117** | **0.106** | **0.115** |
| neutral (ML) | 0.112 | 0.001 | 0.129 |

## Reads

1. **Bombs are deal-driven, not skill-driven** — human bomb/round is flat
   (~0.105–0.110) across *every* decile. Master matches (0.115). Bombing was
   never a candidate differentiator. Settled.
2. **Calls are mildly conservative**, ~19–20% below decile-9 (master calls like
   a decile-8 human). Clean mechanism: the **Call Networks decode by `argmax`**
   (call iff P > 0.5), which under-calls vs a 14.5% human base rate. **Near-free
   lever**: lower the call decision threshold below 0.5 to lift call rate toward
   decile-9 — no RL. Modest EV; worth a cheap A/B.
3. **No gross pathology** — across all measured behaviors master ≈ decile-9
   human. The owner-beats-it gap is sub-behavioral.

## Caveats

- Success rates deliberately NOT compared: manifest `round_outcome` is net round
  score (team0−team1), not "caller went out first" (the ML metric) — a faithful
  comparison needs the replay out-order.
- `trick_win` / `out_first` are degenerate (≡0.25) in self-play and need
  head-to-head profiling; `slam_rate` has a capture bug (out_order reset by
  `_finalise_round` before the round-end step is observed). Neither affects the
  call/bomb conclusions above.

## Bearing on the plan

- **PPO rationale narrows to one**: sharpen decision quality past the human-modal
  line (Bayes-ceiling argument), NOT "add a missing behavior." Consistent with
  [BC is label-bound, not capacity-bound](2026-06-02-bc-is-label-bound-not-capacity-bound.md)
  — imitation is at its intrinsic ceiling.
- **Belief/search** (opponent modeling) is the other untested suspect for the
  sub-behavioral gap; orthogonal to PPO.
- Cheap side-quest: recalibrate the Call-Network decision threshold.
