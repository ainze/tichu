---
id: "006"
title: "Featurizer + canonical action space (versioned)"
type: AFK
blocked_by: ["002"]
stories: [27, 28, 29]
---

## What to build

Define the two versioned contracts that everything ML-side depends on: the featurizer and the canonical action space. Both must be pinned to every saved checkpoint; mismatches must fail loudly at load time.

**Canonical action space.** A single ordered list of every possible action in Tichu: play-combination actions (one entry per legal combination type × content), pass action, Tichu call, Grand Tichu call, schupfen (pass-card) choices, wish-rank choices (2–A), dragon-give-direction choices. The exact size will be determined during enumeration (PRD estimates 5,000–10,000). The ordering is fixed and must never change within a version. Persisted alongside every checkpoint. Loading a checkpoint with a mismatched action-space version raises immediately.

Round-trip test: for every action in the canonical list, `decode(encode(action)) == action`.

**Featurizer.** A pure function `featurize(private_state: PrivateState) -> np.ndarray` with no I/O, no globals, no time-dependence. Its output shape is fixed for a given version. Version is persisted alongside every checkpoint; loading a checkpoint with a mismatched featurizer version raises immediately.

Purity test: calling `featurize` on the same `PrivateState` twice (in the same process and across processes) returns byte-identical arrays.

Version pinning test: constructing a model with version `"v1"` and attempting to load a checkpoint saved with version `"v2"` raises a `VersionMismatchError` with a message identifying which version string collided.

**Note on sequencing.** This slice depends only on `#002` (the engine), not on the parser. The featurizer is defined against `PrivateState` — parser integration (stamping `featurizer_version` into Parquet records) happens in `#007`.

## Acceptance criteria

- [ ] Canonical action space is fully enumerated and its size is logged at import time
- [ ] Action space version is a string constant persisted with checkpoints; mismatch raises `VersionMismatchError`
- [ ] `featurize(private_state)` returns a fixed-shape `ndarray` regardless of game phase
- [ ] Featurizer version is a string constant persisted with checkpoints; mismatch raises `VersionMismatchError`
- [ ] Purity test passes: byte-identical output across two processes given the same input
- [ ] Round-trip test passes for every action in the canonical list
- [ ] Version pinning test confirms mismatched versions raise, not silently corrupt

## Blocked by

- [#002 Rules engine](002-rules-engine.md)
