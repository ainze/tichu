"""Torch-free per-decision-type constants.

`HEAD_LOGIT_DIMS` lives here (not in `bc/heads.py`) so the BC dataset
readers (`bc/dataset.py`, `bc/materialised.py`) and the parse pipeline
that builds them (`bsw/to_parquet.py`, the `parse_bsw` CLI) can pull
the constant without dragging `torch` into their import graph.

Before this split, importing `parse_bsw` on a server without torch
installed failed at `to_parquet -> bc/dataset -> bc/heads -> import
torch`. parse_bsw doesn't need torch — it produces parquet shards and
optionally a memmap bundle — so the import-time dependency was a
leak, not a real requirement.

`bc/heads.py` re-exports `HEAD_LOGIT_DIMS` from here unchanged so all
existing `from tichu_training.bc.heads import HEAD_LOGIT_DIMS` call
sites continue to work without modification.
"""

from collections import OrderedDict

from tichu_training.action_space import ACTION_SPACE_SIZE


# Per-decision-type head output sizes. Schupfen is NOT here — it is
# served by a standalone Schupfen Network per ADR-0012, not a BC head.
HEAD_LOGIT_DIMS: "OrderedDict[str, int]" = OrderedDict([
    ("play", ACTION_SPACE_SIZE),
    ("wish", 14),
    ("dragon_assignment", 2),
])
