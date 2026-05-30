# ADR-0017: Featurizer v4 replaces the 1,809-dim `trick_top_combo` one-hot with a 50-dim union-of-fields layout

- **Status:** Accepted
- **Date:** 2026-05-30
- **Related:** [ADR-0011](0011-bc-training-replay-on-the-fly.md), [ADR-0014](0014-pre-featurise-bc-corpus.md), [ADR-0015](0015-featurizer-v3-drop-leaky-and-redundant-sections.md), [ADR-0016](0016-consolidate-parse-and-materialise-into-one-replay.md)

## Context

After ADR-0015, the v3 featurizer emits 1,983 floats per `PrivateState`.
One section now dominates the size, and a count of its constituent
slots reveals that almost all of them carry no useful signal at
inference time.

Per-section breakdown of the v3 vector:

| Section | Dims | % | Notes |
| --- | ---: | ---: | --- |
| `trick_top_combo` | 1,809 | 91.2% | One-hot over the v1 Action Space — the current top combination |
| `own_hand` + `seen_cards` | 112 | 5.6% | 56 dims each, multi-hot card slots |
| 11 other sections | 62 | 3.1% | hand_sizes, scores, callers, wish, phase, etc. |
| **Total** | **1,983** | 100% | |

Three findings drove the v4 redesign.

**Finding 1 — ~1,223 of the 1,809 `trick_top_combo` slots (68%) are
phoenix-variants.** Each `(combination-shape, rank, length)` tuple in
the v1 Action Space generates as many phoenix-position variants as the
combo has slots: a 13-card straight from rank 2 produces 12 distinct
indices for "phoenix at position k" plus 1 natural, so 13 indices for
one straight shape. Summed across all straights, pair-steps,
full-houses, and rank-groups, the phoenix-variants outnumber the
naturals by ~2:1.

**Finding 2 — at most one phoenix-variant can ever fire per round.**
There is one Phoenix per game; once played, it's gone for the rest of
the round. So each individual phoenix-variant slot activates roughly
1 in tens of thousands of training rows. Most v3 trick_top_combo
weights are touched by gradient flow at a near-zero rate.

**Finding 3 — phoenix position is not beat-relevant.** Beatability of
a straight depends on `(start_rank, length)`; the rules don't care
where inside the straight the Phoenix sits. Same for pair-steps and
full-houses. The marginal information a phoenix-variant slot carries
over `(intent_kind, primary_rank, length) + phoenix_used` is the
position — and the position is irrelevant for every downstream
decision.

**Finding 4 — storage gates the corpus path.** Per ADR-0014, the
materialised bundle is the throughput path. At v3:

- 100k subset, 53.5M rows × (1983 × 4 B features + 1809 B raw mask +
  13 B meta) ≈ **499 GiB**.
- Full corpus (~2.4M games, ~1.5B decisions, linear extrapolation) ≈
  **14 TiB**.

The 100k bundle is 7.8× the box's 64 GB RAM, so the OS page cache
can't hold the working set; first-epoch throughput is dominated by
cold-page faults on `play_features.dat` (~396 GiB by itself). The
full corpus is well over 4 TB NVMe headroom, forcing NAS / tiered
storage in which every cold page becomes a network seek.

## Decision

**Bump `FEATURIZER_VERSION` from `"v3"` to `"v4"`. Replace the 1,809-dim
`trick_top_combo` one-hot with a 50-dim union-of-fields encoding.**

| Subfield | Dims | Slot convention |
| --- | ---: | --- |
| `intent_kind` | 8 | Single, Pair, Triple, FullHouse, PairStep, Straight, FourBomb, SFBomb (matches action_space canonical order) |
| `primary_rank` | 15 | slot 0 = Mahjong (rank 1); slots 1..13 = ranks 2..14; slot 14 = Dragon |
| `secondary_rank` | 13 | FullHouse pair_rank only; slots 0..12 = ranks 2..14 |
| `length` | 13 | PairStep / Straight / SFBomb; slots 0..12 = lengths 2..14 |
| `phoenix_used` | 1 | 1 iff Phoenix participates in the combo |
| **Total** | **50** | |

Phoenix-half-rank semantics (engine rank 1.5 for Phoenix-as-lead, N+0.5
for Phoenix-following rank N) are absorbed into the existing fields:
`primary_rank = floor(rank)` + `phoenix_used = 1`. No half-rank slot
needed.

Other featurizer sections are unchanged. The v4 vector is **224 dims**
(1,983 − 1,809 + 50), per-row at f32 **896 B** (8.85× compression on
features). Section order is unchanged so the rest of the cursor
arithmetic stays mechanically identical.

**The Action Space stays at v1.** This is a featurizer-only change.
The play head still outputs 1,809 logits — `trick_top_combo` reused
the Action Space enumeration for cheap input-side mapping; that link
is the one being severed. The play head, the legal-action mask, and
the Resolver are untouched.

## Rationale

1. **Storage drop and what it unlocks.** At v4 the same workloads land
   at:

   | Layout | 100k subset (~53.5M dec) | Full 2.4M corpus (~1.5B dec) |
   | --- | ---: | ---: |
   | v3 raw f32 + raw mask | ~499 GiB | ~14 TiB |
   | v4 raw f32 + raw mask | **~137 GiB** | ~3.3 TiB |

   - **100k at v4 fits in 64 GB RAM** for the feature portion alone
     (~46 GiB of features). After a one-time read-through the OS page
     cache holds the working set; cold-page faults stop dominating
     first-epoch throughput.
   - **Full corpus at v4 fits on a single 4 TB NVMe** with headroom.
     No tiered storage; no NAS-or-external-disk seeks. Quantisation
     remains tractable as a follow-up (int8 + bit-packed mask would
     bring full corpus to ~0.8 TB) but is no longer load-bearing.

2. **The dropped phoenix-variant slots carry ~zero information.** A v3
   slot like `PlayStraight(start_rank=2, length=13, phoenix_position=7)`
   fires when one specific 13-card straight is the top and Phoenix
   substitutes at offset 7 — a vanishingly rare configuration. Its
   marginal information over `(intent_kind=Straight, primary_rank=1,
   length=12, phoenix_used=1)` is the phoenix offset, which has no
   beat-relevance (any 13-card straight starting at rank 2 wins by
   `(start_rank, length)` match, regardless of phoenix position). The
   model loses representational capacity it cannot productively use.

3. **`phoenix_used=1` is not redundant with `seen_cards[phoenix]=1`.**
   The two diverge when Phoenix was burned in an earlier trick:
   `seen_cards[phoenix]=1, phoenix_used=0` says "phoenix is gone but
   not in the current top." `phoenix_used=1` adds the specific signal
   "the current top contains the phoenix" — useful for the model when
   reasoning about whether the opponent burned a powerful card on
   this particular trick.

4. **Cross-kind weight sharing on `primary_rank`.** The union-of-fields
   layout (over a per-intent-kind discriminated union) shares the 15
   `primary_rank` neurons across all intent kinds. The trunk learns
   "rank 7 is involved" once, not 8 times. This is the central
   architectural reason `primary_rank` is 15 dims rather than 15 ×
   8 = 120 dims, and the reason scalar `comparison_rank` was rejected
   — the one-hot keeps rank-specific weights free to learn while
   still sharing them across shape contexts.

5. **Featurizer-only diff keeps blast radius small.** Action Space
   stays v1; play head dims stay 1,809; the legal-mask universe is
   unchanged; the Resolver is unchanged. The change is contained to
   `src/tichu_training/featurizer.py` plus parity tests for each
   intent kind. v3 checkpoints fail to load against v4 code with a
   clear `VersionMismatchError` from the existing checkpoint pin; the
   same goes for v3 materialised bundles. No engine code changes, no
   `PublicState` schema changes, no ADR-0008 replay-validation impact.

## Consequences

- **v3 BC checkpoints are not loadable against v4 code.** The
  trunk's input-projection layer changes from `Linear(2047, 1024)` to
  `Linear(288, 1024)` (where 64 is the SkillEmbedding dim, unchanged).
  The `Checkpoint`-level `featurizer_version` pin catches this with a
  legible error before the shape mismatch surfaces.

- **v3 materialised bundles are not loadable against v4 code.**
  `MemmapBCDataset.__init__` raises `VersionMismatchError` when the
  bundle's `featurizer_version` doesn't match the live constant. The
  100k v3 bundle (`data/materialised_100k`) has been archived; v4
  re-materialisation produces `data/parquet_100k_feat_v4` and
  `data/materialised_100k_v4` as parallel canonical paths.

- **The play head's output is unchanged.** The model still emits
  1,809 logits, the legal-mask is still in v1-Action-Space
  coordinates, and the Resolver still maps Intents to Concrete Actions
  exactly as before. `ACTION_SPACE_VERSION` stays `"v1"`.

- **TrueSkill / SkillEmbedding wiring is unchanged.** The per-row
  `skill_decile` baked into the materialised bundle's `META_DTYPE` is
  featurizer-version-independent; the same `ratings_*.parquet` files
  feed v4 re-materialisation without recomputation.

- **The v3 baseline lives only in archived artefacts.** The active
  100k training run (`data/runs/bc_full_100k_memmap`, checkpoint
  `step_e000_b00025000.bin`) is frozen as part of the v3 archive. Any
  v3 ↔ v4 comparison reads from the archived run's logged loss/acc
  trajectories — no live v3 retraining is planned.

- **Featurizer test surface grows but doesn't shift.** Per-intent-kind
  slot assertions replace the v3 generic "section fires on non-empty
  top" check. 12 explicit tests covering Single (natural, Mahjong,
  Dragon, Phoenix-as-lead, Phoenix-following), Pair (natural, with
  phoenix), Triple, FullHouse (phoenix in triple, phoenix in pair),
  PairStep, Straight (mahjong-led, natural with phoenix), FourBomb,
  StraightFlushBomb. The four version-pin / dim-total tests bump from
  v3 to v4 values.

- **Bundle-naming convention.** Per-bundle and per-parquet
  directories carry a `_v<N>` suffix matching `FEATURIZER_VERSION`.
  This is unchanged from the operational pattern; restated here so
  the convention survives the v3 archive removing the previous live
  example.

## Rejected alternatives

- **Per-intent-kind discriminated-union encoding** (8 disjoint
  sub-blocks, ~50 dims total but no field overlap). Rejected: weaker
  cross-kind weight sharing on rank. A "rank-7 single" and a
  "rank-7 pair" would fire entirely different neurons; the trunk
  would have to relearn rank semantics in each shape context. The
  union-of-fields layout costs the same dim count and lets the trunk
  share "the rank that matters here is 7" once.

- **Dense scalar encoding** (rank as `(r-1)/14 ∈ [0,1]`, length as
  `(L-2)/12`, etc., ~10 dims total). Rejected: asymmetric risk. The
  one-hot lets the first linear layer learn rank-specific weights
  freely; the scalar forces ordinal structure that may or may not
  match the true rank-utility curve. Cheap dim savings are not worth
  potentially measurable BC quality loss.

- **Keep `suit[4]` in the trick_top_combo section.** Rejected: zero
  beat-relevant signal and zero scoring-relevant signal across every
  intent kind. Singles/Pairs/Triples beat by rank, ignoring suit.
  StraightFlushBomb beats by `(length, start_rank)`, ignoring suit
  (a longer SF bomb of any suit beats; a same-length SF bomb of
  higher start_rank beats). 5-point and 10-point cards are
  suit-independent in scoring. The only theoretical use is
  card-counting — which `seen_cards` already covers exactly.

- **Keep `phoenix_position` as a per-kind offset slot.** Rejected:
  position is not beat-relevant for any of FullHouse, PairStep, or
  Straight (the kinds where it would conceivably matter). The
  "phoenix was burned this trick" signal that survives is captured by
  `phoenix_used[1]`. The position itself only matters to the player
  holding the phoenix — which is the player whose hand is already
  encoded in `own_hand`.

- **Explicit `intent_kind[9]` lead / no-top slot.** Rejected: lead
  state is inferable from the rest of the vector (no current top →
  all-zero `intent_kind` → distinct from any populated kind). The
  rare unrepresentable-top case (FullHouse with Phoenix in both
  triple and pair, Phoenix-following-Mahjong) collapses with lead in
  v4 exactly as it does in v3; this is acceptable noise.

- **`primary_rank[14]` instead of `[15]`.** Rejected: leaves no
  clean slot for Dragon. Either Dragon collides with rank-14 ("rank
  14, phoenix_used=0" can't mean both Dragon and natural Ace) or a
  separate `is_dragon` bit pulls in the same dim cost with uglier
  semantics.

- **`secondary_rank[14]` instead of `[13]`.** Rejected: pair_rank
  can never be Mahjong or Dragon (rules — neither special can
  participate in a pair). The extra slot would be a permanent zero.

- **`length[14]` instead of `[13]`.** Same — lengths are 2..14, not
  1..14. One permanently-zero slot.

- **Engine-level changes** (`played_by` per-card attribution,
  `tricks_won` per-player count). Deferred — same rationale as
  ADR-0015 §5. These would require `PublicState` schema changes and
  ADR-level work. v4 is featurizer-only.

- **Quantisation in v4** (bf16 / int8 features, bit-packed mask).
  Out of scope. v4's 224-dim raw f32 is sufficient for the full
  corpus at NVMe. Quantisation remains tractable as a follow-up ADR
  if needed for further compression (full corpus at int8 + packed
  mask would land near ~0.8 TB) but the v4 design alone closes the
  load-bearing gap that ADR-0015 left open.
