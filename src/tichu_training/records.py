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
import pyarrow.parquet as pq

from tichu_training.checkpoint import VersionMismatchError


def load_shards(
    directory: str | Path,
    decision_type: str,
    *,
    expected_featurizer_version: str,
    expected_action_space_version: str,
) -> pa.Table:
    """Concatenate `{decision_type}_*.parquet` shards from `directory`.

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

    tables = [pq.read_table(p) for p in shards]
    table = pa.concat_tables(tables)

    _verify_version_column(table, "featurizer_version", expected_featurizer_version)
    _verify_version_column(table, "action_space_version", expected_action_space_version)
    return table


def _verify_version_column(table: pa.Table, column: str, expected: str) -> None:
    seen = {v for v in table.column(column).to_pylist() if v is not None}
    if seen != {expected}:
        raise VersionMismatchError(
            f"{column} mismatch: shards contain {sorted(seen)!r}, "
            f"loader expected '{expected}'"
        )
