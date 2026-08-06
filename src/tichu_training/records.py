"""Read per-decision-type training shards with version-mismatch detection.

Shards on disk follow the naming convention `{decision_type}_NNNNN.parquet`
(NNNNN = zero-padded shard index, currently always `00000`; multi-shard
splits can fill in higher suffixes later without changing this contract).

`load_shards` concatenates shards in sorted name order and verifies that
the `featurizer_version` and `action_space_version` columns contain only
the values the caller expects. A mismatch raises `VersionMismatchError`
identifying the column, the stored values seen, and the expected value.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tichu_training.checkpoint import VersionMismatchError


def load_shards(
    directory: str | Path,
    decision_type: str,
    *,
    expected_featurizer_version: str | None,
    expected_action_space_version: str | None,
    columns: list[str] | None = None,
) -> pa.Table:
    """Concatenate `{decision_type}_*.parquet` shards from `directory`.

    `columns` projects the read. The full corpus is ~1.45e9 rows x 15 columns;
    callers that need two id columns (e.g. the BC manifest build) must say so or
    the read alone exhausts RAM. The version columns are always read for the pin
    and dropped again if the caller did not ask for them.

    An expected version of `None` WAIVES that pin. Use it only where the column
    is vestigial: the shards' `featurizer_version` describes the `state` /
    `legal_actions_mask` columns, which are 100% null under ADR-0011's
    replay-on-the-fly design, so pinning it would force a multi-hour BSW re-parse
    on every featurizer bump for a field that documents nothing. The pin that
    actually protects the corpus is on the materialised BUNDLE manifest, which
    `materialise()` stamps with the live FEATURIZER_VERSION.

    Raises `FileNotFoundError` if no shards match.
    Raises `VersionMismatchError` if any row's version columns disagree with
    the caller's expectation.
    """
    directory = Path(directory)
    shards = sorted(directory.glob(f"{decision_type}_*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no shards matched {decision_type}_*.parquet under {directory}"
        )

    _VERSION_COLUMNS = ("featurizer_version", "action_space_version")
    read_columns = None
    if columns is not None:
        read_columns = list(dict.fromkeys([*columns, *_VERSION_COLUMNS]))

    tables = [pq.read_table(p, columns=read_columns) for p in shards]
    table = pa.concat_tables(tables)

    _verify_version_column(table, "featurizer_version", expected_featurizer_version)
    _verify_version_column(table, "action_space_version", expected_action_space_version)

    if columns is not None:
        table = table.select(list(columns))
    return table


def _verify_version_column(
    table: pa.Table, column: str, expected: str | None,
) -> None:
    if expected is None:
        return  # pin waived by the caller; see `load_shards`
    # `unique` on the ChunkedArray, NOT `to_pylist()`: the play shard holds
    # ~1.45e9 rows, and materialising that many Python strings to build a set of
    # (in practice) ONE value dies with MemoryError.
    seen = {
        v for v in pc.unique(table.column(column)).to_pylist() if v is not None
    }
    if seen != {expected}:
        raise VersionMismatchError(
            f"{column} mismatch: shards contain {sorted(seen)!r}, "
            f"loader expected '{expected}'"
        )
