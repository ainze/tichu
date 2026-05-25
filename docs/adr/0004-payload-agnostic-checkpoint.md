# ADR-0004: Checkpoint format is payload-agnostic

- **Status:** Accepted
- **Date:** 2026-05-25
- **Related:** [ADR-0001](0001-trunk-architecture.md), [ADR-0002](0002-intent-level-action-space.md)

## Context

Four distinct kinds of trained artefact share the on-disk Checkpoint format
defined in `tichu_training/checkpoint.py`:

1. **BC Checkpoint** — Trunk + four BC Heads.
2. **Refined Checkpoint** — byte-identical layout to a BC Checkpoint; produced
   by AWR Refine starting from a BC Checkpoint.
3. **Call Network Checkpoint** — one small Tichu-or-Grand-Tichu network.
4. **Belief Checkpoint** — the opponent-hand prediction network.

The Checkpoint format records exactly two version strings —
`featurizer_version` and `action_space_version` — plus opaque payload
bytes. It does **not** record which of the four kinds the payload is.
A reader that loads a Call Network Checkpoint expecting a BC payload will
deserialise garbage and crash inside PyTorch, not at the Checkpoint layer.

## Decision

**Checkpoint stays payload-agnostic. The loader's expected-class dictates
how the bytes are parsed.** The header records *only* the two version pins
that affect cross-checkpoint compatibility, never the payload shape.

## Rationale

1. **Compatibility versioning is orthogonal to payload identity.** A BC
   Checkpoint and a Refined Checkpoint share both versions *and* the exact
   payload shape — they are interchangeable by construction. Encoding a
   "kind" field would force them to disagree, breaking the hot-swap
   property that the Difficulty Spec relies on (Difficulty `hard` →
   `master` is a Checkpoint path swap; the ML Agent loader is identical).
2. **Filename + caller intent already disambiguate kind.** A Call Network
   Checkpoint is loaded by `train_calls` or by the ML Agent's call-network
   slot — never confusable with a BC Checkpoint in practice. The cost of
   a mistaken load is a loud PyTorch deserialisation error, not silently
   wrong predictions.
3. **Version pins protect against the actually-dangerous failure mode.**
   The failure that would silently corrupt predictions is a stale
   featurizer or stale action space producing plausible-looking
   nonsense. The header catches both at load time. Payload-kind mismatches
   are loud anyway.
4. **The format stays small and stable.** Adding a `kind` field
   retroactively would either break every existing checkpoint (forcing
   re-training of everything in flight) or require a tagged-union
   migration that adds complexity without protecting against any
   silent-failure mode.

## Consequences

- Code that loads a Checkpoint **must know which payload kind it expects**
  before calling `Checkpoint.load(...)`. There is no introspection step.
- Refined Checkpoints can be passed wherever BC Checkpoints are accepted,
  by design. This is the substrate of ADR-0003's Difficulty Spec.
- If a fifth Checkpoint payload kind is ever added (e.g. a phase-2 search
  policy), it must come with its own loader that uses the same Checkpoint
  format — not extend the format itself.
- Diagnostics tooling that wants to inspect "what's in this file"
  cannot read the kind from the header. It must try a payload parser
  speculatively, or rely on the filename convention.

## Rejected alternative

**Add a `payload_kind` field to the Checkpoint header.** Rejected because
it would break the BC/Refined byte-compatibility (forcing an explicit
"BC vs AWR" distinction that the rest of the system intentionally avoids)
and because the silent-failure mode it would prevent does not exist —
mismatched payload deserialisation already errors loudly inside PyTorch.
