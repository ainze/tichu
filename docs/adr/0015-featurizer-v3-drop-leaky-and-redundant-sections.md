# ADR-0015: Featurizer v3 drops `play_history`, `schupfen_received`, and `phoenix_played`

- **Status:** Accepted
- **Date:** 2026-05-29
- **Related:** [ADR-0011](0011-bc-training-replay-on-the-fly.md), [ADR-0012](0012-schupfen-is-a-standalone-network.md), [ADR-0013](0013-parquet-schema-versioned-by-directory.md), [ADR-0014](0014-pre-featurise-bc-corpus.md)

## Context

The v2 featurizer emits 16,624 floats per `PrivateState`. Three sections
dominate the size; one of them also turns out to leak information the
game's rules say the player doesn't have.

Per-section breakdown of the v2 vector:

| Section | Dims | % | Notes |
| --- | ---: | ---: | --- |
| `play_history` | 14,472 | 87.0% | Last 8 plays in the **current trick**, one-hot over `ACTION_SPACE_SIZE=1809` |
| `trick_top_combo` | 1,809 | 10.9% | One-hot: current top combination on the trick |
| `schupfen_received` | 168 | 1.0% | 3 directions × 56 cards: cards passed to this player by each opponent / partner |
| `own_hand` + `seen_cards` + 12 others | 175 | 1.1% | The rest of the vector |
| **Total** | **16,624** | 100% | |

Three findings drove the v3 redesign.

**Finding 1 — `play_history` is structurally redundant with `trick_top_combo`.**
`trick.top_combination` is defined as `trick.plays[-1].combination`
([src/tichu_engine/state.py:51](../../src/tichu_engine/state.py)). The
last play *is* the top combination — they're the same datum. The
marginal information `play_history` adds over `trick_top_combo` alone
is the **within-trick sequence** (which combination came before the
top, who responded to whom). A typical trick is 4-8 plays; this is
narrow temporal context, not round-wide history.

**Finding 2 — round-wide card visibility is already covered by `seen_cards`.**
The v1 featurizer used `play_history` + `phoenix_played` to expose
which cards had been played. v2 added `seen_cards` (56-dim multi-hot
over `PublicState.played_cards_this_round`), which gives the
round-wide card-counting signal directly. `play_history`'s
round-history function was already obsolete in v2.

**Finding 3 — `schupfen_received` is a rules violation.**
The v2 encoding lays out `direction_idx × 56` slots, where
`direction_idx ∈ {previous-seat, partner, next-seat}` is structurally
distinct ([src/tichu_training/featurizer.py:268-289 v2](../../src/tichu_training/featurizer.py)).
This gives the model per-giver attribution of every card it received
during schupfen — information a rule-abiding player does not have.
The receiving player sees three new cards land in their hand; the
rules do not let them know which opponent vs their partner gave them
which card. A model trained on this feature learns to use the
attribution signal, then either (a) fails to deploy honestly (the
feature can't be populated at inference without cheating), or (b) is
deployed and **the agent cheats in production**.

**Finding 4 — `phoenix_played` is redundant with `seen_cards`.**
The 1-bit flag "is Phoenix visible in the current trick" duplicates
information already in `seen_cards[phoenix_slot]` once Phoenix has
been played in the round.

## Decision

**Bump `FEATURIZER_VERSION` from `"v2"` to `"v3"`. Drop three sections.**

| Section | v2 dims | v3 dims | Rationale |
| --- | ---: | ---: | --- |
| `play_history` | 14,472 | **0** | Redundant with `trick_top_combo`. Within-trick sequence is not worth 87% of corpus footprint. |
| `schupfen_received` | 168 | **0** | Rules violation — leaks per-giver attribution. |
| `phoenix_played` | 1 | **0** | Redundant with `seen_cards`. |
| Everything else | 1,983 | 1,983 | Unchanged. |
| **Total** | **16,624** | **1,983** | **8.4× smaller** |

No engine changes. No `PublicState` schema changes. No new sections.
Strictly subtractive — the v3 vector is a subsequence of the v2 vector
with three blocks removed and `cursor` arithmetic adjusted.

## Rationale

1. **Storage gates the corpus path.** ADR-0014 adopted pre-featurise
   (β') as an additive training source. At v2 the materialised bundle
   sized at ~4 TB for the 100k-game subset and ~13 TB for the full
   2.4M-game corpus — the full-corpus number was the binding "is this
   architecturally viable" question. At v3 the same workloads are
   ~492 GB and ~1.48 TB respectively (raw f32, no quantisation, no
   sparsity). **The full 2.4M-game corpus now fits on a single 2 TB
   NVMe.** Storage stops being the question.

2. **`play_history`'s information content is small relative to its
   cost.** Round-wide card visibility lives in `seen_cards`; the
   current top combination lives in `trick_top_combo`; the per-trick
   pass set lives in `trick_passes`. What `play_history` adds is the
   *sequence* of plays within the current trick — which combination
   was led, which was the first response, who's been in the
   escalation chain. For BC specifically, the human policies in the
   training data made decisions using this signal, but the dominant
   factors (own hand, top combination, what cards are gone) are
   preserved. The expected BC accuracy regression is bounded; the
   storage saving is 87%. If a measured regression turns out
   significant, a v4 can add back targeted within-trick context
   (e.g., `last_play` as a 1×1809 one-hot, or `last_combination_type`
   as a small categorical) at a tiny fraction of v2's cost.

3. **`schupfen_received` is a defect, not a feature.** It is
   information leakage at training time and a potential cheat at
   inference time. Even if the deployed engine somehow zeros the
   section before featurising at inference (which we have not
   audited), the model trained on it has learned to expect a signal
   that won't be there in honest play — a distribution shift between
   training and deployment. Removing it is unambiguous: the receiver
   already sees their three new cards via `own_hand`; aggregate
   "which cards arrived via schupfen" is the part the player
   legitimately knows, and it's already in `own_hand`.

4. **Featurizer-only diff keeps blast radius small.** The change is
   contained to `src/tichu_training/featurizer.py` plus version-pin
   updates in tests. No engine code changes, no `PublicState`
   schema changes, no ADR-0008 replay-validation impact. v3
   composes cleanly with the materialised bundle path from ADR-0014
   — the bundle's manifest already pins `featurizer_version`, so
   v2 bundles fail to load against v3 code with a clear error.

5. **Better targeted history can come later if needed.** Pre-v3
   discussions explored adding `played_by` (per-card → player
   attribution, 56 × 4 = 224 dims) and `tricks_won` (4 dims).
   Both would require **engine changes** — `PublicState` currently
   stores `played_cards_this_round` as a `frozenset` (no
   attribution) and has no per-player trick-count field. Per the
   ADR-0008 replay-validation contract, adding fields to
   `PublicState` triggers ADR-level work. Deferring those keeps v3
   focused on the dominant storage lever and the rules-fix.

## Consequences

- **v2 BC checkpoints are not loadable against v3 code.** The
  `Checkpoint`-level `featurizer_version` pin catches this. Operators
  must re-train BC from scratch against v3. Models that were trained
  using `schupfen_received` were structurally dependent on a leaked
  signal; the new model breaks that dependency.

- **v2 materialised bundles are not loadable against v3 code.**
  `MemmapBCDataset.__init__` raises `VersionMismatchError` when the
  bundle's `featurizer_version` doesn't match the live constant. The
  17 GB v2 pilot bundle and any partial v3-targeted v2 bundle become
  scratch — re-materialise.

- **Existing v2-stamped parquet shards remain readable as a
  manifest.** Per ADR-0011, parquet is the validated `(game_id,
  round_id)` manifest, not the feature data. The `featurizer_version`
  column in parquet is a defensive pin against accidentally training
  on shards generated by a different featurizer; in practice the
  shards do not contain features, so the pin's only role is the
  defensive guarantee. Operators training under v3 will need to
  override or refresh the pin — out of scope for this ADR but
  flagged as the immediate next operational task.

- **`tichu_inference/ml_agent.py` becomes safer.** The deployed agent
  no longer has access to the leaked schupfen-attribution signal
  even structurally — there is no v3 slot to populate. Whatever
  cheat the v2 agent may have exhibited (still to be quantified)
  cannot recur in v3.

- **Test fixtures that hard-coded `featurizer_version="v1"` for
  dummy artifacts** broke and were fixed inline. The fixtures now
  track the live `FEATURIZER_VERSION` / `ACTION_SPACE_VERSION`
  constants. Tests that explicitly probe version-mismatch behaviour
  retain their hard-coded mismatch values (e.g. `"v999"`).

- **The `played_by` / `tricks_won` engine extensions remain
  available** as a v4 if measured BC quality on v3 regresses below a
  threshold the operator considers acceptable. Both would land
  behind a future ADR that scopes the `PublicState` schema impact.

## Rejected alternatives

- **Keep `schupfen_received` but zero it out at training time.**
  Rejected: still emits the leaked-shaped slot in the deployed
  agent's input, which the agent could in principle be retrained
  to exploit if the section ever becomes nonzero. Cleaner to remove
  the slot entirely.

- **Keep `play_history` with a smaller window (e.g., 2 instead of 8).**
  Rejected: even at 2 the section is 2 × 1,809 = 3,618 dims =
  ~22% of v2 cost. The win from going to 8 → 2 is significant but
  smaller than the win from 8 → 0, and the rationale that
  `trick_top_combo` already encodes the *last* play applies
  identically. If post-deployment evaluation shows a regression,
  a targeted `last_play` (1 × 1,809) addition is cleaner than
  partial-window preservation.

- **Engine-level `played_by` / `tricks_won` in v3.** Deferred — see
  rationale #5. Featurizer-only v3 is the minimum-commitment path
  that unlocks ADR-0014's full-corpus target.

- **Quantisation (bf16 / int8) in v3.** Out of scope. v3's 1,983-dim
  raw f32 already meets ADR-0014's full-corpus target on a single
  2 TB drive. Quantisation can be layered later as an orthogonal
  optimisation.
