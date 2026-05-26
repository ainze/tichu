# ADR-0012: Schupfen is a standalone network, not a BC head

- **Status:** Accepted
- **Date:** 2026-05-26
- **Related:** [ADR-0007](0007-calls-are-standalone-networks.md), [ADR-0011](0011-bc-training-replay-on-the-fly.md)

## Context

[heads.py](../../src/tichu_training/bc/heads.py) declares
`HEAD_LOGIT_DIMS["schupfen"] = 3`, intended as a "3-way per card" head
for the Schupfen Decision. Two structural problems block this from
training cleanly under the multi-head shared-trunk pattern:

1. **A Schupfen Decision is intrinsically structured.** Each player
   assigns three of fourteen cards to three labeled destinations
   (`to_next`, `to_partner`, `to_previous`). It is not "pick one int
   from K" — it is "pick three distinct cards and assign three
   labels". The shared trunk + single linear head produces a single
   K-way distribution per forward pass; that does not natively express
   the joint constraint.

2. **The featurizer is decision-context-blind to the subject card.**
   [featurizer.py](../../src/tichu_training/featurizer.py) encodes the
   pre-Schupfen hand, the public state, and the schupfen-received
   slots, but carries no signal indicating which specific card the
   model is being asked about. Two BCExamples emitted from the same
   ParsedAction (one per card-given-away) would have identical
   features and different targets — unlearnable.

Three options were considered:

- **B1.** Restructure the schupfen head to a multi-output shape
  (56 cards × 4 directions = 224 logits) and change `BCExample.target`
  from `int` to a per-card vector. Loss = masked multi-output cross-
  entropy. Pure shared-trunk approach; one forward pass per Decision.
- **B2.** Carve schupfen out into a standalone network, mirroring the
  ADR-0007 treatment of Tichu / Grand-Tichu calls. Drop `"schupfen"`
  from `HEAD_LOGIT_DIMS`; new `bc/schupfen_model.py`; new
  `train_schupfen` CLI; the schupfen parquet shard feeds the
  standalone trainer instead of the BC trainer.
- **B3.** Inject a per-call "subject card" signal into the featurizer
  and emit 14 BCExamples per ParsedAction with 4-way targets. Bumps
  `FEATURIZER_VERSION`. Inference would call the schupfen head 14×
  per Decision and resolve the "pick 3 distinct cards for 3 distinct
  destinations" constraint heuristically.

## Decision

**Schupfen is served by a standalone Schupfen Network, not a BC head.**

Concrete shape:

1. **Drop `"schupfen"` from `HEAD_LOGIT_DIMS`.** BCModel goes from
   four heads to three: `play`, `wish`, `dragon_assignment`. The
   `schupfen` parquet shard is *not* consumed by `train_bc`.

2. **New module `tichu_training/bc/schupfen_model.py`.** Defines a
   `SchupfenNetwork` consuming the same `FEATURIZER_OUTPUT_DIM` input
   plus Skill Embedding. Output shape and decoding strategy are
   deferred to the schupfen implementation pass — candidates include
   multi-output 56×4, autoregressive 3-step decoder, or
   pointer-network. None of those choices touch the BC trunk or
   featurizer.

3. **New CLI `train_schupfen`** consumes the `schupfen_00000.parquet`
   shard (manifest only, per ADR-0011's archive-driven pattern). Emits
   a Schupfen Checkpoint, byte-shape-compatible with the existing
   `Checkpoint` container per ADR-0004 (featurizer_version +
   action_space_version + payload bytes).

4. **ML Agent loads four Checkpoints** instead of three: one BC (or
   Refined) + Tichu Call + Grand-Tichu Call + Schupfen. The Difficulty
   Spec gains a `schupfen_path` entry mirroring the existing
   `tichu_call_path` / `grand_tichu_call_path` shape.

5. **The 3 `SchupfenDirection` intents in the Action Space remain.**
   They are the vocabulary the Schupfen Network's output decoder will
   resolve to; carrying them in the canonical Action Space keeps the
   single-source-of-truth for "an Intent" intact.

## Rationale

1. **The shared trunk handles single-discrete-decision heads cleanly;
   structured prediction is the wrong shape for it.** Play, wish, and
   dragon_assignment all map to "pick one int from K". Schupfen
   genuinely does not. Forcing it on (B1) requires invasively
   reshaping `BCExample.target` from `int` to `np.ndarray` and a
   matching loss-function refactor, which contaminates the
   straightforward three-head BC pipeline.
2. **ADR-0007 already established the precedent.** Tichu / Grand-Tichu
   calls were carved out because their decision shape didn't fit.
   Schupfen's mismatch is at least as large — it is not even a single
   discrete decision. Applying the same precedent is consistent, not
   novel.
3. **No version bumps.** Featurizer stays at `v1`; Action Space stays
   at `v1`. B3 would have bumped `FEATURIZER_VERSION`, invalidating
   every Checkpoint that ever ships. B1 arguably bumps
   `ACTION_SPACE_VERSION` (the 3 SchupfenDirection intents become 224
   (card, direction) intents). B2 keeps both pins intact.
4. **Schupfen training defers naturally.** The BC smoke run can ship a
   3-head BC Checkpoint that plays the in-trick decisions; Schupfen
   gets a deliberate design pass once the smoke pipeline is proven.
   Schupfen play strength is second-order: a random schupfen +
   well-trained in-trick play already beats a baseline.
5. **The cost is one more network at inference, not architectural.**
   ML Agent already loads three Checkpoints (BC + 2 Call Networks).
   Four is the same load pattern; the Difficulty Spec gains one path
   field. No new architectural surface.

## Consequences

- `BCModel`'s `heads: nn.ModuleDict` shrinks from four entries to
  three. Any code path that iterated `HEAD_LOGIT_DIMS` expecting
  exactly four heads needs to handle three.
- `ParquetBCDataset.__iter__` filters parquet rows to `decision_type
  ∈ {"play", "wish", "dragon_assignment"}`. The `schupfen` shard is
  still produced by `parse_bsw` (the schema does not change) but is
  read by `train_schupfen`, not `train_bc`.
- ADR-0011 §3's promise of "BC consumes four of six parquet decision
  types" becomes "three of six". The Schupfen ParsedAction 14×
  expansion described in ADR-0011 is also superseded — schupfen
  ParsedActions are now expanded (or not) by `train_schupfen`'s own
  dataset shape, not by `ParquetBCDataset`.
- `legal_mask` in `tichu_training/action_space.py` (also from ADR-0011)
  dispatches on three decision types now, not four. K ∈ {1809, 14, 2}.
- CONTEXT.md's "The 6 Decisions" table moves Schupfen from "BC head
  `schupfen`" to a new "Schupfen Network" served-by row.
- ML Agent (`tichu_inference/ml_agent.py`) gains a `SchupfenNetwork`
  load path. The Difficulty Spec gains `schupfen_path`. Both deferred
  until the Schupfen Network is implemented; until then the ML Agent
  falls back to a heuristic for Schupfen Decisions.
- The CONTEXT.md term "Call Network" is overloaded to also cover the
  Schupfen Network, *or* a parallel "Schupfen Network" term is
  introduced. Choosing the latter keeps each Decision's network
  searchable by name.

## Rejected alternatives

- **B1. Multi-output 56×4 head on the shared trunk.** Rejected:
  contaminates the BC trainer's single-int-target shape, forces a
  multi-output loss path that no other head uses, and arguably bumps
  ACTION_SPACE_VERSION. The complexity is borne by every BC head's
  code path, not just schupfen's.
- **B3. Per-card BCExample with subject-card in features.** Rejected:
  bumps FEATURIZER_VERSION, invalidating every Checkpoint at the
  first iteration of a versioning change that should be done
  sparingly. Also pushes a hard joint-constraint problem (three
  distinct cards for three distinct destinations) onto a heuristic
  inference-time resolver — exactly the design smell that ADR-0007
  flagged for Tichu calls.
