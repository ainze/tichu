# ADR-001: Shared trunk architecture — Large MLP over engineered features

- **Status:** Accepted
- **Date:** 2026-05-15

## Context

The PRD calls for a shared trunk that is reused across the behavioural-cloning
policy, the Tichu-call networks, the belief model, and the eventual offline-RL
refinement step. Two candidates were considered:

1. **Large MLP** over the v1 featurizer's engineered features
   (16,568 floats; see `tichu_training/featurizer.py`).
2. **Transformer** over a token sequence representing trick history, with the
   current public state attached as a CLS-style summary token.

A benchmark on a held-out 50K-decision subset was discussed in the issue but
was skipped in favour of an engineering decision so that downstream issues
(#009, #010, #012, #013) are not blocked. The risk this trades off — picking
the wrong trunk — is acceptable because the chosen architecture is the
lower-variance, lower-risk option and can be replaced by a transformer in a
follow-up if the BC plateau looks like a representational limit rather than
a data or label-noise limit.

## Decision

**Use a large MLP over the v1 engineered features as the shared trunk for
every downstream training task.**

Concrete shape (target, may be tuned in #009):

- Input: 16,568-float vector from `featurize(PrivateState)`.
- 4 residual hidden blocks, width 1024, GELU + LayerNorm.
- Trunk output: 512-dim representation handed to per-task heads.

Per-task heads are independent linear/2-layer MLPs.

## Rationale

1. **The featurizer already encodes structured history.** Sections 11 and 13
   (trick top combo + last 8 plays in the current trick) are one-hot
   encoded against the v1 action space. Most of what a transformer would
   compute from raw history is already pre-flattened. A transformer's main
   structural advantage is therefore partially obviated for Tichu, where
   the history window is bounded (≤ 8 plays / current trick, ≤ 14 cards /
   hand).
2. **DouZero precedent.** DouZero solves Doudizhu — a comparable trick-taking
   imperfect-information card game — with MLP-class architectures plus an
   action-conditional Q-network. That paper demonstrates that MLPs scale to
   billions of game frames on a 3-player setting without a transformer.
3. **Inference latency.** The PRD targets a real-time inference service.
   Forward-pass cost on a width-1024 MLP is ~1ms on a single CPU thread;
   a transformer over even a 30-token history is an order of magnitude
   slower without quantisation.
4. **Iteration speed.** #009 onwards are gated on the trunk being trainable.
   An MLP is simpler to debug, has predictable convergence properties, and
   does not require attention-related tuning (LR warmup, gradient clip,
   pre-LN vs post-LN, etc.). Faster path through #009→#013.
5. **Lower variance of training outcomes.** MLP training behaviour on tabular
   features is well-understood. Failure modes (overfit, capacity-bound) are
   easy to diagnose with held-out cross-entropy.

## Consequences

- #009 (BC training pipeline) instantiates the MLP trunk above.
- #010, #012, #013 reuse the same trunk weights (or a fine-tunable copy)
  via the `tichu_training.checkpoint` contract introduced in #006c.
- If BC accuracy plateaus below an acceptable level *and* failure analysis
  points at a representational ceiling rather than data quality, the
  follow-up is a focused transformer experiment on the same train/val split.
- The benchmark in #008 is deferred but not cancelled — it would be the
  natural first experiment to run if/when the plateau is observed.

## Rejected alternative

**Transformer over trick history.** Higher capability ceiling but:

- Most of its advantage is already captured by the v1 featurizer's flat
  history encoding.
- Substantially heavier inference cost.
- Higher tuning risk on a moderate (50K–290M decision) dataset.

Revisit if the MLP trunk underperforms a literature-derived baseline.
