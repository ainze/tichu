# ADR-0021: Belief trains on a replay-derived bundle

- **Status:** Accepted
- **Date:** 2026-05-31
- **Related:** [ADR-0020](0020-one-parse-pass-many-task-bundles.md) (the pipeline that emits this bundle), [ADR-0014](0014-pre-featurise-bc-corpus.md) / [ADR-0019](0019-bit-pack-materialised-bundle.md) (bundle format + bit-packing), [ADR-0018](0018-tichu-call-featurises-at-first-non-pass-play.md) (precedent: choosing a featurise moment), [ADR-0012](0012-schupfen-is-a-standalone-network.md) (relative-seat direction convention)

## Context

CONTEXT.md's **Belief Model** entry has long stated it is *"trained on
the BSW corpus using … hidden hands as labels."* But the only dataset
that ships
([`belief/dataset.py`](../../src/tichu_training/belief/dataset.py)) is
`SyntheticBeliefDataset` — pure RNG. The real-data path the glossary
promised was never implemented; the synthetic dataset is a smoke
placeholder, exactly as `SyntheticCallDataset` / `SyntheticSchupfenDataset`
were before their Parquet datasets existed.

[ADR-0020](0020-one-parse-pass-many-task-bundles.md) makes building it
now nearly free. Belief is the one task that is **both replay-bound and
cost-free to capture during the consolidated pass**: at every Play
Decision the replay already holds **every seat's Hand** in the
`pre_decision_state`, so the labels (which opponent holds which card) are
literally sitting in the engine state we already compute. Building belief
later means re-running a full-corpus replay (~hours) purely to recover
hidden Hands we are otherwise discarding right now. This is the strongest
"one replay, many outputs" case of any task.

`BeliefModel`
([`belief/model.py`](../../src/tichu_training/belief/model.py)) is
already `feature_dim`-agnostic: a flat feature vector → `(3 opponents,
56 cards)` occupancy logits, with a masked binary cross-entropy loss.
Nothing in the model presupposes the synthetic input.

## Decision

**Add a real, replay-derived belief training source, materialised
alongside the BC bundle in the same parse pass
([ADR-0020](0020-one-parse-pass-many-task-bundles.md)). Belief is gated
behind explicit `--bundle-tasks belief` until its reader and a real
`train_belief` path are proven.**

1. **Input = the same 224-dim Feature Vector.** Belief consumes
   `featurize(PrivateState)` for the acting (current) Player — the same
   vector the BC `play` example already computes at that decision. No
   belief-specific featurizer. This shares the feat_bits/feat_cont
   packing primitive and makes belief's only new per-row work the
   label/mask extraction (the features are reused, not recomputed). No
   leakage: `own_hand` is legitimate conditioning, everything else is
   public.

2. **Sample at every Play Decision, acting Player's view.** One belief
   example per Play Decision. Labels are read from the **same
   `pre_decision_state`** that produced the features — so they are always
   fresh to the featurise moment (no staleness by construction). Other
   Decision types (wish / dragon / calls) are not sampled — play
   decisions carry the card-counting signal and align with the reused
   feature vector.

3. **Capture all; filter later.** Early-Trick states (the
   low-information opening that recurs at the start of *every* Round,
   since each Round re-deals) carry near-uniform belief targets. Rather
   than drop them at **write** time — irreversible, recoverable only by
   another full replay, and tunable only once Phase-2 calibration can be
   measured — every Play-Decision example is emitted with a tiny
   `cards_played u1` meta field (`56 − sum(hand_sizes)`). The
   low-information cut becomes a **read-time filter** in the belief
   trainer: reversible, measurable, no data lost.

4. **Relative-seat opponent ordering.** The `(3,56)` label grid indexes
   opponents as **next / partner / previous** — the same convention
   **Schupfen** uses — so the model is seat-invariant (it learns
   relative-position beliefs, not absolute-seat artifacts). Labels come
   from `pre_state.hands` (all four Hands present at every
   `pre_decision_state`); the three non-acting seats are taken in this
   relative order.

5. **Packing.** Labels `(3,56)` bool → flatten 168 bits → `np.packbits`
   → 21 B/row; reader unpacks and reshapes, manifest records the `(3,56)`
   shape. The **mask is `(56,)`, not `(3,56)`**: every unplayed,
   non-own card sits in *some* opponent's hand, so "unknown" is a
   card-level fact identical across the three opponents. Stored as 7 B/row
   and broadcast to `(3,56)` in the reader — the truthful representation,
   saving ~0.75 GB on the one large new bundle. Mask semantics for v1 =
   "not in own Hand and not played" (matching the existing model /
   synthetic contract).

6. **A real `MemmapBeliefDataset` reader ships with the bundle**, so the
   bundle is not write-only and the round-trip is proven, even while
   `train_belief`'s production wiring and Phase-1-vs-Phase-2 use remain
   deferred.

## Rationale

1. **The labels are free exactly here and expensive everywhere else.**
   The replay already materialises all four Hands at each Play Decision;
   extracting the opponent grid is a handful of array writes against
   state we computed anyway. Deferring belief forfeits this and re-incurs
   the full replay later.
2. **Reusing the Feature Vector reuses the BC computation.** Sampling at
   Play Decisions for the acting seat means belief's features are the
   identical vector the BC `play` example produced — zero extra
   featurise, one feature codec across all four bundles.
3. **Capture-all + read-time filter respects what is and isn't yet
   measurable.** The right low-information cutoff is an empirical
   question about belief calibration vs. cards-played; baking a heuristic
   at write time pre-commits an unmeasured choice and is irreversible
   without another replay.
4. **Relative-seat ordering reuses an established convention** (ADR-0012
   schupfen directions) and gives the model translation invariance for
   free.
5. **Card-level `(56,)` mask is the honest shape.** Storing it `(3,56)`
   would triple a redundant array on the largest new bundle.

## Consequences

- **The belief bundle is play-scale**, the largest of the new bundles
  (~the BC `play` row count). Packed it is ~(67 B features + 21 B labels
  + 7 B mask + meta)/row — a few GB at the live scale, decisively not
  "tiny" like calls/schupfen. Cheap to *produce* (features reused), real
  on disk.
- **Belief is opt-in only** (`--bundle-tasks belief`), never in `all`,
  until the reader + real `train_belief` path are proven
  ([ADR-0020](0020-one-parse-pass-many-task-bundles.md)).
- **`SyntheticBeliefDataset` stays** as the smoke source; CONTEXT.md's
  Belief Model entry is updated to name the relative-seat ordering, the
  per-Play-Decision sampling, and the synthetic-placeholder distinction.
- **Schupfen-passed-card knowledge is out of scope for v1.** The acting
  Player knows which 3 cards they Schupfen'd away; the v1 mask ignores
  this (the featurizer may already encode it, so the model can learn it).
  A future ADR may tighten the mask if it measurably helps.
- **The cutoff is a future tuning ADR, not a re-materialise.** Because
  `cards_played` is carried per row, experimenting with where to start
  belief training needs no new replay.

## Rejected alternatives

- **Keep belief synthetic-only / defer entirely.** Rejected: the
  hidden-Hand labels are free *only* during this pass; deferring re-incurs
  a full-corpus replay and leaves CONTEXT.md's promise unmet.
- **A belief-specific input encoding.** Rejected: forks the feature codec,
  forfeits the shared packing and the BC-feature reuse, for no shown gain
  — the 224-dim vector already encodes `seen_cards` / `out_order`, the
  core card-counting signal.
- **A hard write-time cutoff (drop the early Trick / "first round").**
  Rejected: irreversible, recoverable only by re-replay, and pre-commits
  an unmeasured calibration choice. Superseded by the `cards_played`
  read-time filter. (Also a glossary trap: each Round re-deals, so the
  low-information phase is the opening of *every* Round, not Game-Round
  #1.)
- **Subsample states (every-Nth / high-skill-only) at write time.**
  Rejected for v1 for the same reversibility reason; a follow-up ADR can
  add it if the play-scale footprint binds, mirroring ADR-0014's deferred
  partial-materialisation.
- **`(3,56)` mask.** Rejected: redundant (identical across opponents) on
  the largest new bundle.
