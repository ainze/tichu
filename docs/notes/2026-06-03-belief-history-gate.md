# 2026-06-03 — Belief history-input gate (ADR-0028 ablation)

## TL;DR

The decline-history channel **helps** (B-core > A > naive on BCE, acc@0.5, and
which-opponent top-1), and the play-time channel **hurts** (B-full < B-core) —
so the ADR-0028 design lands at **B-core, drop H4**. *But* the absolute signal
is weak: trained belief beats a trivial hand-size predictor by only ~2–4 pp
top-1, in **every** round-progress bucket, with no regime of dramatic advantage.
At this scale (3k games, hidden 256, 8 epochs, not converged) explicit belief
adds little the policy can't already infer from `hand_sizes` + `seen_cards`. A
scale check is warranted before betting the expensive policy-retrain on belief.

## Setup

3000-game subset, B-full belief bundle (307-dim), three trainings selecting the
nested tier prefix (A=:224, B-core=:251, B-full=:307). Floor = the max-entropy
predictor that knows only opponent hand sizes
([scripts/belief_naive_baseline.py](../../scripts/belief_naive_baseline.py)).

## Results

Final-epoch training metrics:

| | masked BCE | acc@0.5 |
|---|---|---|
| naive | 0.5599 | 0.6975 |
| A (224) | 0.5276 | 0.7067 |
| **B-core (251)** | **0.5213** | **0.7082** |
| B-full (307) | 0.5229 | 0.7079 |

Eval over 100k examples, incl. the discriminating which-opponent top-1:

| | BCE | acc@0.5 | top-1 | top-1 by round (open/mid/late/end) |
|---|---|---|---|---|
| naive | 0.556 | 0.699 | 0.472 | 0.365 / 0.479 / 0.648 / 0.813 |
| A | 0.541 | 0.708 | 0.502 | 0.397 / 0.50 / 0.66 / 0.83 |
| **B-core** | **0.539** | **0.709** | **0.505** | **0.408 / 0.509 / 0.669 / 0.830** |

## Reads

1. **Ablation direction is clean and monotone**: B-core > A > naive on all
   metrics; B-full < B-core. **Decision: belief input = B-core; H4 (play-time)
   rejected** — it was player-anonymous and silent on Passes, as feared.
2. **Naive is strongly round-dependent** (0.365 → 0.813): late-game hand sizes
   nearly determine the holder, so the floor is high exactly where you'd hope
   belief helps — leaving little headroom.
3. **Belief's lift over naive is small and ~uniform** (+1.7 to +4.3 pp),
   *largest in the opening* (where hand sizes are equal) and smallest at the end.
   There is **no regime where explicit belief dominates** the hand-size
   heuristic the policy can already compute. The decline channel's own
   contribution (B-core − A) is ~0.3 pp aggregate.

## Caveats

- **Scale**: 3k games, hidden 256, 8 epochs, loss still dropping. Undertrained /
  undersized; a larger run could lift the absolute numbers. The ablation
  *direction* is scale-robust; the *absolute strength* is not yet established.
- **Metric proxy**: aggregate "which opponent holds a card" is an imperfect
  proxy for play value, which lives in specific high-stakes inferences (can opp
  X beat my Y / bomb?), not average holder accuracy. The decisive test is
  extrinsic (belief → policy → Tournament).

## Scale check (20k games, hidden 512, 12 epochs, streaming) — PLATEAU

Ran the pre-committed scale check (10.76M examples, 2× capacity, +50% epochs)
to decide scale-limited vs near-ceiling. Decision rule going in: late-game
top-1 0.669 → ~0.74+ = integrate; within ~1–2 pp = pivot.

| B-core | BCE | acc@0.5 | top-1 | by round (open/mid/late/end) |
|---|---|---|---|---|
| gate (3k, h256, 8ep) | 0.539 | 0.709 | 0.505 | 0.408 / 0.509 / 0.669 / 0.830 |
| **scale (20k, h512, 12ep)** | 0.534 | 0.712 | 0.510 | 0.410 / 0.513 / **0.674** / 0.835 |
| naive (scale bundle) | 0.556 | 0.699 | 0.473 | 0.366 / 0.478 / 0.646 / 0.810 |

**10× data + 2× capacity + more epochs bought +0.5 pp top-1 (+0.5 pp late).**
Squarely in the "plateau" band. Training loss was already flat by epoch 11.

**Verdict: belief is near-ceiling, NOT scale-limited.** It tops out ~3–4 pp above
the hand-size floor the policy can already compute (`hand_sizes` + `seen_cards`),
and barely beats it late-game (0.674 vs 0.646) where it should matter most —
because the residual opponent-hand uncertainty in Tichu is largely irreducible
from the observable features. This mirrors the BC label-ceiling result: more
scale doesn't move a signal that isn't there.

## Bearing on the plan

- ADR-0028 → Accepted for B-core; H4 dropped. The belief *infrastructure* (history
  features, streaming trainer) is sound and stays for its Phase-2 **search** role
  (ADR-0021) — it just is not the play-strength lever.
- **Per the decision rule: pivot.** Explicit belief does not beat hand-size counting
  by enough to justify the policy-retrain; the opponent-modeling hypothesis is
  weakened (caveat: top-1 accuracy ≠ play-value, but the scale-invariance is a
  strong signal). The surviving lever for play strength is **PPO sharpening** of
  the sub-behavioral decision quality (Bayes-ceiling rationale), which every
  diagnostic so far has left standing.
