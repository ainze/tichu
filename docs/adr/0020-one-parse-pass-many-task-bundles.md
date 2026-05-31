# ADR-0020: One parse pass, many task bundles

- **Status:** Accepted
- **Date:** 2026-05-31
- **Related:** [ADR-0016](0016-consolidate-parse-and-materialise-into-one-replay.md) (consolidated BC into the replay pass — this generalises it), [ADR-0014](0014-pre-featurise-bc-corpus.md) (the bundle format), [ADR-0019](0019-bit-pack-materialised-bundle.md) (bit-packing), [ADR-0007](0007-calls-are-standalone-networks.md) (calls standalone), [ADR-0012](0012-schupfen-is-a-standalone-network.md) (schupfen standalone), [ADR-0018](0018-tichu-call-featurises-at-first-non-pass-play.md) (tichu featurise moment), [ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md) (belief's bundle, decided alongside this one)

## Context

[ADR-0016](0016-consolidate-parse-and-materialise-into-one-replay.md)
folded the **BC** bundle write into `parse_bsw`'s existing replay pass:
the round that is replayed for Parquet validation also emits the
featurised `BCExample` stream, materialised to a `MemmapBCDataset`
bundle. One replay, both outputs — eliminating the second full replay
that `materialise_bc` used to pay.

But BC's three heads (`play` / `wish` / `dragon_assignment`) are only
three of the **six** Decision types. The other replay-bound and
parse-bound training tasks each still run their **own** archive pass:

- **Calls** (`train_calls`, [`bc/call_training.py`](../../src/tichu_training/bc/call_training.py)).
  Tichu featurises at each seat's first non-Pass Play
  ([ADR-0018](0018-tichu-call-featurises-at-first-non-pass-play.md)) and
  **replays the round** (`replay_round(..., early_stop=...)`,
  `:175`). Grand-Tichu featurises a *synthetic deal-time* `GameState`
  from `pre_deal_hands` — **no replay** (`:145`).
- **Schupfen** (`train_schupfen`,
  [`bc/schupfen_training.py`](../../src/tichu_training/bc/schupfen_training.py)).
  Featurises a *synthetic pre-schupfen* `GameState` from `start_hands` —
  **no replay** (`_schupfen_examples_for_round`, `:223`).
- **Belief** (`train_belief`,
  [`belief/dataset.py`](../../src/tichu_training/belief/dataset.py)).
  Synthetic-only today; the real replay-derived path is decided in
  [ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md) and is
  genuinely replay-bound (labels = hidden Hands at each Play Decision).

So the framing "one replay, many outputs" is imprecise: only **Tichu
Call** and **Belief** are replay-bound. Grand-Tichu and Schupfen are
**parse-bound** — their trainers never step the engine. The honest
generalisation is **one parse pass, many outputs**, with two tiers of
saving:

1. **Replay-reuse** (Tichu, Belief): the consolidated full replay
   already visits the states these tasks need (Tichu's first-non-Pass-Play
   pre-states; Belief's per-Play-Decision states with all four Hands
   present). Folding them in eliminates a *second full replay* — the
   ~75 % dominator measured in
   [ADR-0014](0014-pre-featurise-bc-corpus.md) /
   [ADR-0016](0016-consolidate-parse-and-materialise-into-one-replay.md).
2. **Parse-reuse / pipeline-unification** (Grand-Tichu, Schupfen): no
   replay to save, but folding them in saves the re-parse + re-featurise
   and collapses three CLIs into one pass. Marginal CPU, real ergonomics.

## Decision

**Extend the consolidated pass so a single `parse_bsw` run emits a
materialised bundle per replay-bound *and* parse-bound training task,
one bundle directory per consuming trainer.**

### Bundle topology — one directory per trainer

Four bundle families, grouped by the trainer that loads them, **not** by
Decision type:

| Bundle | Types inside | Consumed by |
| --- | --- | --- |
| BC (existing) | `play`, `wish`, `dragon_assignment` | `train_bc` |
| Calls | `call_tichu`, `call_grand_tichu` | `train_calls` |
| Schupfen | `schupfen` | `train_schupfen` |
| Belief | `belief` | `train_belief` |

Each directory is a **self-contained, independently-versioned bundle**
(its own `manifest.json`, `schema_version`, and featurizer / action-space
pins). The Calls bundle keeps both call types under one directory — one
`train_calls` invocation loads both — mirroring how the BC bundle holds
three types. The replay/parse stays shared (the win); each trainer's
storage and version contract stays decoupled (the hygiene that
[ADR-0007](0007-calls-are-standalone-networks.md) /
[ADR-0012](0012-schupfen-is-a-standalone-network.md) demand — calls and
schupfen have independent lifecycles and must not be coupled under the
BC manifest's single `schema_version`).

### CLI — root directory + per-task subdirs

`parse_bsw --bundle-out-dir ROOT` becomes a **parent** directory; each
selected task writes a self-contained bundle to `ROOT/{bc,calls,schupfen,belief}/`.
A new `--bundle-tasks` flag selects the subset:

- **Default `all`** = `bc`, `calls`, `schupfen`.
- **Belief is a valid task but gated *out* of `all`** ([ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md)):
  it is play-scale and Phase-2-only, so it is materialised only when named
  explicitly via `--bundle-tasks belief` — never swept in by `all`. (Implemented:
  `BeliefBundleWriter` / `MemmapBeliefDataset`.)
- The `_v<N>` featurizer-version suffix now lives on the **root**
  (`materialised_100k_v4/`); subdirs are plain names.
- **No root manifest.** Each subdir is already self-describing via its
  triple pin; a root manifest would duplicate the version fields and add
  a fourth drift surface for the byte-identity invariant to police. The
  root stays a plain container.

This redefines the existing `--bundle-out-dir` from "the BC bundle dir"
to "a parent dir" — a deliberate breaking change (see Consequences).

### Factoring — thin shared primitives, not a generic engine

The four tasks share `featurize()` → 224-dim, so the feat_bits/feat_cont
split is identical — but their record shapes diverge sharply:

| | mask | target | extra meta |
| --- | --- | --- | --- |
| BC | `legal_mask` K-wide (packed) | scalar `u2` | `round_outcome`, `game_won` (AWR) |
| Calls | none | binary scalar | — |
| Schupfen | `hand_mask` 56 (packed) | 3-vector of slot ids | — |
| Belief | `(56,)` mask (packed) | `(3,56)` multi-hot (packed) | `cards_played` |

The genuinely common machinery is extracted into one module of **thin
primitives** — `pack_features` / `unpack_features` (driven by
`featurizer.CONTINUOUS_FEATURE_COLUMNS`), `packbits` / `unpackbits`
bool-row helpers (covering every mask/label above uniformly),
`_contiguous_runs`, `_check_pin`, and manifest read/write. Each task
keeps its own small `materialise_X()` + `MemmapXDataset`, composed from
those primitives, owning its column layout and meta dtype.

A **generic schema-driven bundle engine** (a task = a declarative column
spec) was rejected: belief's `(3,56)`-label / `(56,)`-mask shape is the
most divergent and was still in design when this was decided; locking a
generic abstraction around unfinalised shapes is the larger risk, and
YAGNI. Promote to an engine only if a fifth task makes the primitives
painful. (Same posture ADR-0016 took deferring the emit-loop de-dup.)

### Emit reuse — share, don't mirror

ADR-0016 *mirrored* `ParquetBCDataset.__iter__` into
`_emit_bc_examples_for_round` (a hand-kept copy guarded by a byte-identity
test) because sharing touched two hot-path modules. For the **new** tasks
we do the opposite: the consolidated pass calls the **same** per-round
emit function the standalone dataset calls.

- `_schupfen_examples_for_round(parsed_round, …)` — already takes only
  `parsed_round`; called directly from both paths.
- A new `_grand_tichu_examples_for_round(parsed_round, …)` and
  `_tichu_examples_for_round(parsed_round, replay, …)` are factored out
  of `_call_examples_for_game` into a **new module
  [`bc/call_emit.py`](../../src/tichu_training/bc/call_emit.py)**. The
  Tichu variant consumes the *passed-in* `ReplayResult` (the consolidated
  pass's full replay) instead of re-replaying — this is the Tichu
  replay-reuse. Both `ParquetCallDataset` and the consolidated pass
  import from `call_emit.py`.

Putting the factored functions in a *new* module (rather than growing
`call_training.py`) shrinks the surface that the separate in-flight
Tichu-call train/val-split work has to rebase onto.

### Testing — graduated, matched to the emit topology

Because the new tasks share (one emit path, nothing to drift), a full
byte-identity parity test à la BC is largely redundant. Instead:

1. **Per new bundle:** round-trip exactness + on-disk packed-size +
   writer-guard tests, mirroring
   [`tests/training/bc/test_materialised.py`](../../tests/training/bc/test_materialised.py).
2. **One focused Tichu parity test:** consolidated full-replay extraction
   ≡ standalone `ParquetCallDataset` early-stop extraction, same rounds —
   the one place two different state-derivation routes must agree.
3. **Light "both paths emit the same rows" smoke** for schupfen /
   grand-tichu (same synthetic-state function from both paths; full
   byte-identity is redundant).
4. BC's existing `test_to_bundle.py` byte-identity test stays (BC still
   mirrors).

## Rationale

1. **The replay-reuse win is real where it exists, and named honestly.**
   Tichu and Belief stop paying a second full replay (the ~75 % CPU
   dominator). Grand-Tichu and Schupfen never replayed, so their win is
   parse-reuse + one-pass ergonomics, not replay elimination — and the
   ADR says so rather than overclaiming.
2. **One directory per trainer keeps the storage contract aligned with
   the network lifecycle.** ADR-0007 and ADR-0012 carved calls and
   schupfen out as standalone networks with independent training cadence
   and versioning. Their bundles inherit that independence: a featurizer
   bump invalidates all four uniformly (they share `featurize()`), but a
   schema change to one trainer's layout does not churn the others.
3. **Thin primitives extract the duplication that actually exists**
   (feature packing, bit-packing, the pin/manifest plumbing) without
   forcing four differently-shaped records through one premature
   abstraction.
4. **Share-not-mirror is strictly better than BC's situation** for the
   new tasks: single source of truth, no new parity tests, and it is the
   *only* way Tichu gets replay-reuse — a mirror that re-replays would
   defeat the purpose.

## Consequences

- **`--bundle-out-dir` is now a parent dir (breaking).** Existing
  runbooks passing `--bundle-out-dir DIR` now get `DIR/bc/`. The BC
  byte-identity test (`test_to_bundle.py`) moves to compare `ROOT/bc/`.
  Accepted for the ergonomic single-pass tree.
- **The Calls and Schupfen bundles are tiny** (single-digit GB at full
  corpus — wish/dragon-scale volumes). The **Belief bundle is not**: it
  is play-scale (~the BC `play` row count) — see
  [ADR-0021](0021-belief-trains-on-a-replay-derived-bundle.md).
- **`materialise_bc`, `ParquetCallDataset`, `ParquetSchupfenDataset` are
  unchanged as standalone paths.** The consolidated pass is additive and
  opt-in per task; a manifest-only `parse_bsw` run still pays nothing.
- **The in-flight Tichu-call train/val-split work** (`game_id` on
  `CallExample`, `mask_self_tichu_call`, `SECTION_OFFSETS`) collides with
  the `call_emit.py` factor-out. Containing the factored functions in a
  new module minimises the rebase surface, but coordination is still
  required when both land.
- **New-task bundles carry provenance.** `game_id u4` + `round_id u1` are
  added to the calls / schupfen / belief meta (BSW game_ids are
  int-parseable and fit `u4`; ~5 B/row on bundles that are tiny anyway).
  This future-proofs the in-flight calls train/val-split (which splits by
  `game_id`), supports per-game filtering, and aids debugging. **BC's
  stable v2 schema is left untouched** — it never needed `game_id`
  (held-out games are filtered before materialise). Merely carrying
  `game_id` does not commit any reader to a split implementation.

## Rejected alternatives

- **One mega-bundle with a single `TYPE_ORDER` covering all six Decision
  types.** Rejected: couples calls/schupfen/belief under the BC
  manifest's single `schema_version`, defeating the independent
  lifecycles of the standalone networks (ADR-0007 / ADR-0012). A schema
  change to one would invalidate all.
- **One bundle directory per Decision type** (six+ dirs). Rejected:
  fragments the call pair that one `train_calls` invocation always loads
  together, and multiplies manifests for no lifecycle benefit.
- **One root + subdirs with `--bundle-tasks` defaulting to `bc`.**
  Rejected in favour of `all` (implemented tasks) — the point of one pass
  is to emit everything ready in a single replay; the marginal cost of
  calls + schupfen is trivial.
- **Generic schema-driven bundle engine.** Rejected for now (see
  Decision §Factoring): premature abstraction around belief's
  still-in-design shape.
- **Editing ADR-0016 in place.** Rejected: ADR-0016 is Accepted and
  scoped to BC; house style supersedes/extends via new ADRs (cf.
  ADR-0019 revising ADR-0014).
