"""ParquetCallDataset: archive-driven negative synthesis for Call Networks.

ADR-0007 follow-up: the call shards from #007 hold only positive call events
(`call_tichu_*.parquet`, `call_grand_tichu_*.parquet`). The dataset enumerates
all four seats per validated round, marks each as positive (seat called) or
negative (seat did not), featurizes at the call moment, and yields a
`CallExample`.

Tracer bullet: Grand-Tichu path on the sample data. The Grand-Tichu featurise
moment is deal-time (8-card pre_deal_hands), no engine replay needed.
"""

import base64
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.call_training import CallExample, ParquetCallDataset
from tichu_training.checkpoint import VersionMismatchError
from tichu_training.cli.parse_bsw import main as parse_bsw_main
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM, FEATURIZER_VERSION


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def _build_indexed_archive(samples_dir: Path, archive_path: Path) -> None:
    """Build the indexed-blob archive format `tools/compress.py` writes,
    so iter_archive can read it. Mirrors the fixture in test_parquet_dataset.py."""
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
    archive_path.with_suffix(archive_path.suffix + ".idx").write_text(json.dumps(sidecar))


@pytest.fixture
def smoke_setup(tmp_path: Path):
    """End-to-end fixture: archive + parquet shards from the sample data."""
    archive = tmp_path / "sample.zst"
    _build_indexed_archive(_SAMPLES, archive)

    shards_dir = tmp_path / "shards"
    rc = parse_bsw_main(["--archive", str(archive), "--output", str(shards_dir)])
    assert rc == 0
    return {"archive": archive, "shards_dir": shards_dir, "tmp_path": tmp_path}


def test_cli_runs_parquet_path_end_to_end(smoke_setup, tmp_path: Path):
    """Drives the `train_calls` CLI through the parquet path on the sample
    archive. Confirms the `NotImplementedError` branch is gone and both
    networks train to a final checkpoint."""
    import csv
    import yaml

    from tichu_training.cli.train_calls import main

    config = {
        "dataset": "parquet",
        "dataset_kwargs": {
            "shards_dir": str(smoke_setup["shards_dir"]),
            "archive_path": str(smoke_setup["archive"]),
            "expected_featurizer_version": FEATURIZER_VERSION,
            "expected_action_space_version": ACTION_SPACE_VERSION,
        },
        "model": {"hidden": 16, "skill_dim": 4},
        "learning_rate": 1e-3,
        "batch_size": 8,
        "epochs": 1,
        "seed": 0,
    }
    config_path = tmp_path / "calls_parquet_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_dir = tmp_path / "run"

    rc = main(["--config", str(config_path), "--run-dir", str(run_dir)])
    assert rc == 0

    for tag in ("grand", "tichu"):
        log_path = run_dir / f"{tag}_step.csv"
        assert log_path.exists(), f"missing {log_path}"
        rows = list(csv.DictReader(log_path.open("r", encoding="utf-8")))
        assert rows, f"{log_path} has no rows"
        assert (run_dir / "checkpoints" / f"{tag}_final.bin").exists()


def test_tichu_positive_count_matches_parsed_callers_who_played(smoke_setup):
    """Total positive Tichu examples = sum across validated rounds of
    `|{seat in tichu_callers : seat played a non-Pass Play this round}|`.

    The "who played" filter matters because a caller who never played a
    non-Pass Play (vanishingly rare — partner swept the round before they
    got a turn) is silently dropped by the C-wide rule. Independently
    parses + replays to compute the expectation."""
    from tichu_training.bsw.archive import iter_archive
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.replay import replay_round

    ds = ParquetCallDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        call_type="tichu",
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    expected_positives = 0
    for stem, text in iter_archive(smoke_setup["archive"]):
        if stem not in ds._manifest:
            continue
        game = parse_tch(text, game_id=stem)
        valid_rounds = ds._manifest[stem]
        for r in game.rounds:
            if r.round_index not in valid_rounds:
                continue
            if not r.tichu_callers:
                continue
            replay = replay_round(r)
            if replay.final_state is None:
                continue
            seats_with_non_pass_play: set[int] = set()
            for (action, _), pre_state in zip(
                replay.decisions, replay.pre_decision_states,
            ):
                if pre_state is None or action.kind != "play":
                    continue
                seats_with_non_pass_play.add(action.player)
            expected_positives += len(r.tichu_callers & seats_with_non_pass_play)
    assert expected_positives > 0, "sample data should contain at least one Tichu call"

    examples = list(ds)
    actual_positives = sum(1 for ex in examples if ex.target == 1)
    assert actual_positives == expected_positives, (
        f"expected {expected_positives} positive Tichu examples (callers who "
        f"made at least one non-Pass Play), got {actual_positives}"
    )


def _example_key(ex):
    return (ex.target, ex.skill_decile, round(ex.sample_weight, 6), ex.features.tobytes())


@pytest.mark.parametrize("call_type", ["grand_tichu", "tichu"])
def test_parallel_workers_match_serial(smoke_setup, call_type):
    """`workers=N` must yield exactly the same multiset of CallExamples as
    `workers=1` — only the per-game order may differ (futures land out of
    order). Validates picklability of the worker + examples, the pool
    initializer, and the round-trip end-to-end."""
    common = dict(
        archive_path=smoke_setup["archive"],
        call_type=call_type,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    serial = list(ParquetCallDataset(smoke_setup["shards_dir"], workers=1, **common))
    parallel = list(ParquetCallDataset(smoke_setup["shards_dir"], workers=2, **common))

    assert serial, "serial produced no examples"
    assert sorted(map(_example_key, serial)) == sorted(map(_example_key, parallel)), (
        f"{call_type}: parallel examples differ from serial "
        f"(serial={len(serial)}, parallel={len(parallel)})"
    )


def test_version_mismatch_raises_at_construction(smoke_setup):
    """Mismatched featurizer_version fires `VersionMismatchError` at
    construction, before any iteration. Same load-time invariant as
    ParquetBCDataset (ADR-0011, ADR-0013)."""
    with pytest.raises(VersionMismatchError):
        ParquetCallDataset(
            smoke_setup["shards_dir"],
            archive_path=smoke_setup["archive"],
            call_type="grand_tichu",
            expected_featurizer_version="UNEXPECTED",
            expected_action_space_version=ACTION_SPACE_VERSION,
        )


def test_ratings_join_populates_skill_decile(smoke_setup, tmp_path: Path):
    """Handles present in the ratings parquet get the joined decile;
    handles missing fall back to the Neutral Skill Decile (=10).
    Confirms the parquet-side join contract from ParquetBCDataset
    carries over to the call adapter."""
    # Pick one handle that appears in the manifest's parsed rounds.
    play_table = pq.read_table(
        smoke_setup["shards_dir"] / "play_00000.parquet",
        columns=["player_handle"],
    )
    handles = {h for h in play_table.column("player_handle").to_pylist() if h}
    target_handle = next(iter(handles))

    ratings_path = tmp_path / "ratings.parquet"
    ratings_table = pa.Table.from_pydict(
        {
            "player_handle": [target_handle],
            "skill_decile": [7],
        },
        schema=pa.schema([
            ("player_handle", pa.string()),
            ("skill_decile", pa.int32()),
        ]),
    )
    pq.write_table(ratings_table, ratings_path)

    ds = ParquetCallDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        call_type="grand_tichu",
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
        ratings_path=ratings_path,
    )
    examples = list(ds)
    assert any(ex.skill_decile == 7 for ex in examples), (
        "expected at least one example to carry the joined decile (7)"
    )
    for ex in examples:
        if ex.skill_decile != 7:
            assert ex.skill_decile == 10, (
                f"non-target handles should fall back to neutral decile 10, "
                f"got {ex.skill_decile}"
            )


def test_tichu_emits_at_most_one_example_per_seat_per_validated_round(smoke_setup):
    """Tichu featurise moment is the seat's first non-Pass Play (per Q2/Q5:
    C-wide symmetric). A seat that never makes a non-Pass Play in a round
    contributes no Tichu example for that round — so the count is bounded
    by 4 × validated_rounds, but may be lower.

    This tracer asserts: examples are emitted, well-formed, and the count
    is in `(0, 4 * manifest_size]`. Positive/negative split is tightened
    in a later test."""
    ds = ParquetCallDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        call_type="tichu",
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    examples = list(ds)
    assert len(examples) > 0, "expected at least one Tichu CallExample"
    assert len(examples) <= 4 * ds.manifest_size, (
        f"Tichu count {len(examples)} exceeds 4 × validated rounds "
        f"({ds.manifest_size})"
    )
    for ex in examples:
        assert isinstance(ex, CallExample)
        assert ex.target in (0, 1), ex.target
        assert ex.features.shape == (FEATURIZER_OUTPUT_DIM,)
        assert 0 <= ex.skill_decile <= 10, ex.skill_decile


def test_grand_tichu_positive_count_matches_parsed_callers(smoke_setup):
    """Tightens the spec: the total number of positive Grand-Tichu examples
    equals the sum of `|grand_tichu_callers|` across every validated round
    in the manifest. Independently parses the sample games to compute the
    expected count, then compares.

    This is the negative-synthesis correctness check — if the dataset ever
    drops a positive, mis-attributes a seat, or emits a positive for a
    non-caller, this fires."""
    from tichu_training.bsw.archive import iter_archive
    from tichu_training.bsw.parser import parse_tch

    ds = ParquetCallDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        call_type="grand_tichu",
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    # Independently compute expected positives from the archive.
    expected_positives = 0
    for stem, text in iter_archive(smoke_setup["archive"]):
        if stem not in ds._manifest:
            continue
        game = parse_tch(text, game_id=stem)
        valid_rounds = ds._manifest[stem]
        for r in game.rounds:
            if r.round_index in valid_rounds:
                expected_positives += len(r.grand_tichu_callers)
    assert expected_positives > 0, "sample data should contain at least one Grand-Tichu call"

    examples = list(ds)
    actual_positives = sum(1 for ex in examples if ex.target == 1)
    assert actual_positives == expected_positives, (
        f"expected {expected_positives} positive Grand-Tichu examples, "
        f"got {actual_positives}"
    )


def test_grand_tichu_emits_one_example_per_seat_per_validated_round(smoke_setup):
    """Tracer bullet: the dataset enumerates all four seats per validated
    round as Grand-Tichu opportunities, with target=1 for seats in
    parsed_round.grand_tichu_callers and target=0 otherwise.

    Asserts the central "negative synthesis" invariant: total examples =
    4 × validated_rounds. Does not yet check that the featurize moment
    is the right one — only that the enumeration happens."""
    ds = ParquetCallDataset(
        smoke_setup["shards_dir"],
        archive_path=smoke_setup["archive"],
        call_type="grand_tichu",
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    examples = list(ds)
    assert len(examples) > 0, "expected at least one CallExample from sample data"
    assert len(examples) == 4 * ds.manifest_size, (
        f"expected 4 examples per validated round ({ds.manifest_size} rounds), "
        f"got {len(examples)}"
    )
    for ex in examples:
        assert isinstance(ex, CallExample)
        assert ex.target in (0, 1), ex.target
        assert ex.features.shape == (FEATURIZER_OUTPUT_DIM,)
        assert 0 <= ex.skill_decile <= 10, ex.skill_decile
