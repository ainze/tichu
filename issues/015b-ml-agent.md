---
id: "015b"
title: "MLAgent — wraps exported TorchScript model with version assertion + fallback"
type: AFK
parent: "015"
blocked_by: ["015a"]
---

## What to build

`tichu_inference.ml_agent.MLAgent` — an `Agent`-compatible adapter that wraps a TorchScript-exported policy.

- Constructor loads the artifact via `tichu_export.torchscript.load_exported(...)` with the trainer's `FEATURIZER_VERSION` + `ACTION_SPACE_VERSION` pinned at the loader. Mismatch raises at construction so it surfaces at service startup, not request time.
- `act(private_state)` runs: featurize → model → mask illegal logits to `-inf` → argmax → decode action index back to engine `Action`.
- **Fallback path**: on any of {NaN logits, all-illegal mask, exception in the inference path, no model loaded}, log an `ERROR`-level message with the offending state hash and return a uniform-random legal action. The game never stalls.
- Optional `rank_actions(private_state)` returns all legal actions sorted by predicted probability (top-5 ready for #011 move-prediction).

## Acceptance criteria

- [ ] Constructed from an exported BC artifact: `act` returns a legal action for a deal-initial state.
- [ ] Outputs match the eager model on the same input (within float tolerance).
- [ ] Fallback fires when the wrapped module raises; result is still a legal action; ERROR log line emitted.
- [ ] Fallback fires when logits contain NaN; same guarantee.
- [ ] `rank_actions` returns a non-empty list whose entries are all legal.

## Blocked by

- [#015a JSON codec](015a-private-state-codec.md)
