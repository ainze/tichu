---
id: "006c"
title: "Checkpoint wrapper + VersionMismatchError"
type: AFK
blocked_by: ["006a", "006b"]
parent: "006"
---

## Parent

[#006 Featurizer + action space](006-featurizer-action-space.md)

## What to build

A thin checkpoint container that pins the featurizer and action-space versions to whatever payload is being persisted, and raises a clear error on mismatch at load time.

**API.** In `tichu_training.checkpoint`:

- `class VersionMismatchError(RuntimeError)` — message must identify which version string collided (featurizer vs action-space) and both values.
- `@dataclass(frozen=True) class Checkpoint` — fields:
  - `featurizer_version: str`
  - `action_space_version: str`
  - `payload: bytes` — opaque blob (the future model weights / table data)
- `Checkpoint.save(path)` — serialize as a small header (JSON metadata) followed by the raw `payload` bytes. Storage format and exact bytes are implementation detail; the only contract is round-trip.
- `Checkpoint.load(path, *, expected_featurizer_version: str | None = None, expected_action_space_version: str | None = None) -> Checkpoint` — read back; if either `expected_*` is provided and does not match the persisted value, raise `VersionMismatchError`.

**Default expected versions.** When `expected_*_version` is `None`, the check is skipped (read-only inspection). Production training code passes the current module constants.

## Acceptance criteria

- [ ] `Checkpoint(...)` saves and loads round-trip with identical fields
- [ ] `load(..., expected_featurizer_version="v1")` succeeds when the stored version is `"v1"`
- [ ] `load(..., expected_featurizer_version="v2")` against a `"v1"` checkpoint raises `VersionMismatchError` whose message contains both `"v1"` and `"v2"` and the word `featurizer`
- [ ] Same for action-space mismatch — message contains both values and the word `action_space` (or `action space`)
- [ ] `load` with neither expected version inspects without raising, returning the persisted `Checkpoint`
- [ ] Payload bytes round-trip byte-identical

## Blocked by

- [#006a Action space](006a-action-space.md)
- [#006b Featurizer](006b-featurizer.md)
