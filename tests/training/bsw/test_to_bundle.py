"""Consolidated ingest: one replay pass writes both the parquet manifest
and the materialised BC bundle.

See the 2026-05-29 pipeline-perf handoff — `parse_bsw` and `materialise_bc`
both replay every round through the engine (~75% of worker CPU). These
tests pin the consolidated single-pass behaviour: `stream_to_parquet` with
`bundle_out_dir` set produces a `MemmapBCDataset`-readable bundle whose
per-type `.dat` files are byte-identical to the legacy
`parquet -> ParallelParquetBCDataset -> materialise()` path.
"""

import base64
import json
from dataclasses import replace
from pathlib import Path

import zstandard as zstd

from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import ParquetBCDataset
from tichu_training.bc.materialised import TYPE_ORDER, MemmapBCDataset, materialise
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.to_parquet import stream_to_parquet
from tichu_training.featurizer import FEATURIZER_VERSION


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"

# Name-sorted == numeric-sorted for these ids. `_build_archive` assigns offsets
# in name-sorted order and `iter_archive` yields in offset order, so feeding
# `stream_to_parquet` the games in this order makes the consolidated path emit
# in the same game order the legacy archive-replay path does — a precondition
# for byte-identical bundles.
_GAME_IDS = ["2417500", "2417501"]


def _sample_games():
    return [
        parse_tch((_SAMPLES / f"{gid}.tch").read_text(encoding="utf-8"), game_id=gid)
        for gid in _GAME_IDS
    ]


def _build_archive(samples_dir: Path, archive_path: Path) -> None:
    """Build a tiny zstd archive in the format produced by tools/compress.py
    (mirrors test_archive.py's helper)."""
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
        "version": 1,
        "level": 3,
        "dict": base64.b64encode(dict_data.as_bytes()).decode("ascii"),
        "files": index,
    }
    archive_path.with_suffix(archive_path.suffix + ".idx").write_text(
        json.dumps(sidecar)
    )


def test_bundle_out_dir_writes_readable_bundle(tmp_path):
    """Tracer bullet: when `bundle_out_dir` is set, the same replay pass that
    writes parquet shards also writes a bundle that `MemmapBCDataset` can
    open and iterate."""
    parquet_dir = tmp_path / "parquet"
    bundle_dir = tmp_path / "bundle"
    stream_to_parquet(_sample_games(), parquet_dir, bundle_out_dir=bundle_dir)

    ds = MemmapBCDataset(bundle_dir)
    examples = list(ds)
    assert len(examples) > 0


def test_consolidated_bundle_is_byte_identical_to_legacy_path(tmp_path):
    """The safety net (handoff §"Tests"): the bundle written by the
    consolidated single-pass path must be byte-identical, per decision type,
    to the bundle produced by the legacy two-pass path —
    `stream_to_parquet` -> `ParquetBCDataset` (archive replay) ->
    `materialise()`. Byte parity is what lets the two implementations
    co-exist during migration without silently drifting.
    """
    # --- Legacy two-pass path ---------------------------------------------
    legacy_parquet = tmp_path / "legacy_parquet"
    stream_to_parquet(_sample_games(), legacy_parquet)
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)
    legacy_ds = ParquetBCDataset(
        legacy_parquet,
        archive_path=archive,
        expected_featurizer_version=FEATURIZER_VERSION,
        expected_action_space_version=ACTION_SPACE_VERSION,
    )
    legacy_bundle = tmp_path / "legacy_bundle"
    materialise(legacy_ds, legacy_bundle)

    # --- Consolidated single-pass path ------------------------------------
    consolidated_parquet = tmp_path / "consolidated_parquet"
    consolidated_bundle = tmp_path / "consolidated_bundle"
    stream_to_parquet(
        _sample_games(), consolidated_parquet,
        bundle_out_dir=consolidated_bundle,
    )

    # --- Byte-identical per-type .dat + order.dat -------------------------
    dat_names = (
        [f"{t}_features.dat" for t in TYPE_ORDER]
        + [f"{t}_legal_mask.dat" for t in TYPE_ORDER]
        + [f"{t}_meta.dat" for t in TYPE_ORDER]
        + ["order.dat"]
    )
    for name in dat_names:
        legacy_file = legacy_bundle / name
        consolidated_file = consolidated_bundle / name
        assert legacy_file.exists(), f"legacy bundle missing {name}"
        assert consolidated_file.exists(), f"consolidated bundle missing {name}"
        assert legacy_file.read_bytes() == consolidated_file.read_bytes(), (
            f"{name} differs between legacy and consolidated bundles"
        )


def test_replay_failed_rounds_contribute_no_bundle_rows(tmp_path):
    """The bundle is built from the same Round-granularity filter as the
    parquet manifest (ADR-0009): only rounds whose engine Ergebnis matches
    BSW's contribute rows. If every round is forced to fail validation, the
    bundle must be empty — a round that failed replay never leaks an example.
    """
    game = parse_tch(
        (_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="2417500",
    )

    # Positive control: the untouched game yields rows.
    good_stats = stream_to_parquet(
        [game], tmp_path / "good_pq", bundle_out_dir=tmp_path / "good_bundle",
    )
    good_total = MemmapBCDataset(tmp_path / "good_bundle").total
    assert good_total > 0
    assert good_stats.rounds_matched > 0

    # Tamper every round's recorded Ergebnis to a value the engine can never
    # produce (a Tichu round's scores never sum to 2), forcing a
    # score_mismatch on every round.
    broken = replace(
        game,
        rounds=[replace(r, ergebnis=(1, 1)) for r in game.rounds],
    )
    bad_bundle = tmp_path / "bad_bundle"
    bad_stats = stream_to_parquet(
        [broken], tmp_path / "bad_pq", bundle_out_dir=bad_bundle,
    )
    assert bad_stats.rounds_matched == 0, "expected every round to fail validation"
    # Assert on the writer's recorded total rather than opening the bundle:
    # an empty bundle is (pre-existing) unreadable by MemmapBCDataset, which
    # is a separate concern from "failed rounds emit nothing".
    bad_manifest = json.loads((bad_bundle / "manifest.json").read_text())
    assert bad_manifest["total"] == 0, (
        "a replay-failed round leaked rows into the bundle"
    )
    assert all(v == 0 for v in bad_manifest["counts"].values())


def test_bundle_is_written_by_streaming_not_full_buffering(tmp_path, monkeypatch):
    """The consolidated path must feed the bundle writer incrementally, so
    peak RAM is bounded by `materialise`'s chunk size rather than by the
    whole corpus. Pinned behaviourally: the writer must start pulling
    BCExamples *before* the game source is fully drained. A design that
    buffers every example into a list and calls `materialise` once at the
    end can only start pulling after the source is exhausted.
    """
    # Four valid games (the two samples, duplicated under distinct ids). All
    # produce rows, so the first example reaches the writer after the first
    # game is processed.
    games = [
        parse_tch((_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="g0"),
        parse_tch((_SAMPLES / "2417501.tch").read_text(encoding="utf-8"), game_id="g1"),
        parse_tch((_SAMPLES / "2417500.tch").read_text(encoding="utf-8"), game_id="g2"),
        parse_tch((_SAMPLES / "2417501.tch").read_text(encoding="utf-8"), game_id="g3"),
    ]

    state = {"games_pulled": 0, "games_pulled_at_first_example": None}

    def counting_source():
        for g in games:
            state["games_pulled"] += 1
            yield g

    import tichu_training.bc.materialised as mat
    real_materialise = mat.materialise

    def spying_materialise(stream, out_dir, **kwargs):
        def watched():
            first = True
            for ex in stream:
                if first:
                    state["games_pulled_at_first_example"] = state["games_pulled"]
                    first = False
                yield ex
        return real_materialise(watched(), out_dir, **kwargs)

    monkeypatch.setattr(mat, "materialise", spying_materialise)

    stream_to_parquet(
        counting_source(), tmp_path / "pq", bundle_out_dir=tmp_path / "bundle",
    )

    assert state["games_pulled_at_first_example"] is not None, (
        "the bundle writer was never fed any examples"
    )
    assert state["games_pulled_at_first_example"] < len(games), (
        "the bundle writer only started pulling after the whole source was "
        "drained — the consolidated path is buffering the full corpus in RAM "
        f"(pulled {state['games_pulled_at_first_example']}/{len(games)} games "
        "before the first example reached the writer)"
    )
