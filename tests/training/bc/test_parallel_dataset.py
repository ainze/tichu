"""ParallelParquetBCDataset must produce the same multiset of BCExamples
as the sequential ParquetBCDataset, ignoring order.

Mirrors `test_workers_path_produces_identical_shards_to_sequential` from
tests/training/bsw/test_archive.py — the parse_bsw multiprocessing
correctness invariant. Here the analogue is BCExample emission.
"""

import base64
import hashlib
import json
from pathlib import Path

import pytest
import zstandard as zstd

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample, ParquetBCDataset
from tichu_training.bc.parallel_dataset import ParallelParquetBCDataset
from tichu_training.cli.parse_bsw import main as parse_bsw_main
from tichu_training.featurizer import FEATURIZER_VERSION


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def _build_indexed_archive(samples_dir: Path, archive_path: Path) -> None:
    files = sorted(samples_dir.glob("*.tch"), key=lambda p: p.name)
    payloads = [p.read_bytes() for p in files]
    dict_data = zstd.train_dictionary(8192, payloads * 50)
    cctx = zstd.ZstdCompressor(dict_data=dict_data, level=3)
    index: dict[str, list[int]] = {}
    offset = 0
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as out:
        for path, payload in zip(files, payloads):
            blob = cctx.compress(payload)
            out.write(blob)
            index[path.name] = [offset, len(blob)]
            offset += len(blob)
    sidecar = {
        "version": 1, "level": 3,
        "dict": base64.b64encode(dict_data.as_bytes()).decode("ascii"),
        "files": index,
    }
    archive_path.with_suffix(archive_path.suffix + ".idx").write_text(
        json.dumps(sidecar),
    )


def _fingerprint(ex: BCExample) -> tuple:
    """Order-independent identity for a BCExample. SHA-256 of the numpy
    payloads keeps the multiset comparison exact without dragging the
    raw arrays into Python set hashing."""
    return (
        ex.decision_type,
        int(ex.target),
        float(ex.sample_weight),
        int(ex.skill_decile),
        float(ex.round_outcome),
        hashlib.sha256(ex.features.tobytes()).hexdigest(),
        hashlib.sha256(ex.legal_mask.tobytes()).hexdigest(),
    )


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    """Module-scoped fixture: building the archive + parquet shards is
    the slow part (sample data is small but `parse_bsw` does the full
    streaming pipeline). One build per test module is enough."""
    tmp = tmp_path_factory.mktemp("bc_parallel")
    archive = tmp / "sample.zst"
    _build_indexed_archive(_SAMPLES, archive)
    shards = tmp / "shards"
    rc = parse_bsw_main(["--archive", str(archive), "--output", str(shards)])
    assert rc == 0
    return {"archive": archive, "shards_dir": shards}


def test_parallel_multiset_matches_sequential(fixture):
    """Parallel iteration produces the same multiset of BCExamples as
    sequential, ignoring order. Pins the architectural invariant:
    sharding by hash(game_id) covers every game exactly once across
    workers."""
    sequential = ParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    parallel = ParallelParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=2,
    )
    seq_examples = sorted(_fingerprint(e) for e in sequential)
    par_examples = sorted(_fingerprint(e) for e in parallel)
    assert len(par_examples) == len(seq_examples)
    assert par_examples == seq_examples


def test_parallel_n_rows_matches_sequential(fixture):
    """`n_rows` is the trainer's tqdm-bound; must match between the two
    implementations so the bar shows the same total either way."""
    sequential = ParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    parallel = ParallelParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=2,
    )
    assert parallel.n_rows == sequential.n_rows


def test_parallel_one_worker_equals_single_threaded(fixture):
    """Sanity check the degenerate case: num_workers=1 still emits the
    same examples. Catches sharding bugs that only manifest with N>=2."""
    sequential = ParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    parallel = ParallelParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=1,
    )
    seq_examples = sorted(_fingerprint(e) for e in sequential)
    par_examples = sorted(_fingerprint(e) for e in parallel)
    assert par_examples == seq_examples


def test_parallel_early_termination_does_not_leak_workers(fixture):
    """When the consumer stops early (e.g. trainer hit --max-examples),
    the dataset's __iter__ must signal workers to stop and join cleanly.
    Read the first 5 examples then break; assert the iteration's
    `finally` clause cleaned up by verifying we can construct + iterate
    a fresh instance immediately afterwards."""
    parallel = ParallelParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=2,
    )
    n = 0
    for _ in parallel:
        n += 1
        if n >= 5:
            break
    # If workers leaked, a second iteration would queue-up against the
    # zombies. Just constructing+iterating again must work cleanly.
    parallel2 = ParallelParquetBCDataset(
        fixture["shards_dir"],
        archive_path=fixture["archive"],
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        num_workers=2,
    )
    second_count = sum(1 for _ in parallel2)
    assert second_count > 0
