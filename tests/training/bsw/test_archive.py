"""`iter_archive` reads zstd archives produced by `tools/compress.py`."""

import base64
import json
from pathlib import Path

import pyarrow.parquet as pq
import zstandard as zstd

from tichu_training.bsw.archive import iter_archive
from tichu_training.cli.parse_bsw import main


_SAMPLES = Path(__file__).resolve().parents[3] / "sample"


def _build_archive(samples_dir: Path, archive_path: Path) -> None:
    """Build a tiny archive in the format produced by tools/compress.py."""
    files = sorted(samples_dir.glob("*.tch"), key=lambda p: p.name)
    payloads = [p.read_bytes() for p in files]
    # train_dictionary likes more samples than two; repeat to satisfy it.
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
    idx_path = archive_path.with_suffix(archive_path.suffix + ".idx")
    idx_path.write_text(json.dumps(sidecar))


def test_iter_archive_yields_game_id_and_text_for_each_tch(tmp_path):
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)

    pairs = dict(iter_archive(archive))

    # Compare on raw bytes-decoded text to bypass platform-dependent
    # universal-newline translation in read_text().
    expected = {p.stem: p.read_bytes().decode("utf-8") for p in _SAMPLES.glob("*.tch")}
    assert pairs == expected


def test_iter_archive_yields_in_offset_order_regardless_of_index_order(tmp_path):
    """Perf invariant: archive reads are sequential. iter_archive must yield in
    offset order even when the index serialises file entries in a different
    order (e.g., alphabetical instead of insertion order)."""
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)

    # Reverse the index dict's key order so insertion order != offset order.
    idx_path = archive.with_suffix(archive.suffix + ".idx")
    sidecar = json.loads(idx_path.read_text())
    sidecar["files"] = dict(reversed(list(sidecar["files"].items())))
    idx_path.write_text(json.dumps(sidecar))

    yielded_ids = [game_id for game_id, _ in iter_archive(archive)]
    # Offsets are assigned in name-sorted order by _build_archive, so the
    # canonical offset order is name-sorted.
    expected_ids = sorted(p.stem for p in _SAMPLES.glob("*.tch"))
    assert yielded_ids == expected_ids


def test_cli_archive_mode_produces_same_shards_as_dir_mode(tmp_path):
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)

    archive_out = tmp_path / "archive_out"
    rc = main(["--archive", str(archive), "--output", str(archive_out)])
    assert rc == 0

    dir_out = tmp_path / "dir_out"
    rc = main(["--input", str(_SAMPLES), "--output", str(dir_out)])
    assert rc == 0

    archive_shards = {p.stem for p in archive_out.glob("*.parquet")}
    dir_shards = {p.stem for p in dir_out.glob("*.parquet")}
    assert archive_shards == dir_shards

    for stem in archive_shards:
        a = pq.read_table(archive_out / f"{stem}.parquet")
        d = pq.read_table(dir_out / f"{stem}.parquet")
        assert a.num_rows == d.num_rows, f"row count differs for {stem}"


def test_cli_rejects_both_input_and_archive(tmp_path):
    """argparse mutex: passing both --input and --archive is a parse error."""
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)
    try:
        rc = main([
            "--input", str(_SAMPLES),
            "--archive", str(archive),
            "--output", str(tmp_path / "out"),
        ])
    except SystemExit as exc:
        rc = exc.code
    assert rc != 0


def test_cli_rejects_missing_archive(tmp_path):
    """--archive pointing at a non-existent file is a clear non-zero exit."""
    rc = main([
        "--archive", str(tmp_path / "does-not-exist.zst"),
        "--output", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_workers_path_produces_identical_shards_to_sequential(tmp_path):
    """`--workers 2` must produce the same parquet rows as the sequential path
    (modulo row ordering — workers complete out of order). Pins the
    correctness invariant of the multiprocessing pool."""
    seq_out = tmp_path / "seq"
    rc = main(["--input", str(_SAMPLES), "--output", str(seq_out)])
    assert rc == 0

    par_out = tmp_path / "par"
    rc = main(["--input", str(_SAMPLES), "--output", str(par_out), "--workers", "2"])
    assert rc == 0

    for shard in seq_out.glob("*.parquet"):
        seq = pq.read_table(shard).to_pydict()
        par = pq.read_table(par_out / shard.name).to_pydict()
        # Same row count.
        assert len(seq["game_id"]) == len(par["game_id"]), shard.name
        # Same multiset of rows (sort by composite key to normalise order).
        def _key(row_idx, columns):
            return (
                columns["game_id"][row_idx], columns["round_id"][row_idx],
                columns["player_handle"][row_idx], columns["action_taken"][row_idx],
            )
        seq_rows = sorted(_key(i, seq) for i in range(len(seq["game_id"])))
        par_rows = sorted(_key(i, par) for i in range(len(par["game_id"])))
        assert seq_rows == par_rows, f"row multiset differs for {shard.name}"


def test_archive_mode_subset_takes_first_n_by_sorted_game_id(tmp_path):
    """--subset N in archive mode mirrors dir-mode semantics: first N entries
    sorted by game_id."""
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)

    output = tmp_path / "out"
    rc = main([
        "--archive", str(archive),
        "--output", str(output),
        "--subset", "1",
    ])
    assert rc == 0

    # Only the lowest-sorted game_id should appear in any shard.
    first_id = sorted(p.stem for p in _SAMPLES.glob("*.tch"))[0]
    for shard in output.glob("*.parquet"):
        table = pq.read_table(shard)
        ids = set(table.column("game_id").to_pylist())
        ids.discard("")  # call shards may be empty
        assert ids <= {first_id}, f"{shard.name} has unexpected game_ids: {ids}"


def test_parse_failures_go_to_parse_failures_txt_not_known_bad(tmp_path):
    """Two failure modes are separated on disk:
       - known_bad_games.txt — games that parsed but had ≥1 replay-failed round
       - parse_failures.txt — games whose .tch couldn't be tokenised at all
    """
    # Build a dir input where one file is a real sample (replay-fails) and one
    # is unparseable garbage (parse-fails).
    src = tmp_path / "mixed"
    src.mkdir()
    real_sample = next(_SAMPLES.glob("*.tch"))
    (src / real_sample.name).write_bytes(real_sample.read_bytes())
    (src / "99999999.tch").write_text("this is not a tichu log file\n", encoding="utf-8")

    output = tmp_path / "out"
    rc = main(["--input", str(src), "--output", str(output)])
    assert rc == 0

    known_bad = (output / "known_bad_games.txt").read_text().split()
    parse_failures = (output / "parse_failures.txt").read_text().split()

    # The real sample parsed but its rounds don't fully match — known-bad.
    assert real_sample.stem in known_bad
    assert real_sample.stem not in parse_failures
    # The garbage file failed to tokenise — parse-failure.
    assert "99999999" in parse_failures
    assert "99999999" not in known_bad


def test_game_id_flag_targets_specific_games_and_dumps_tch(tmp_path):
    """`--game-id ID` debug flag: restricts the source to the specified IDs
    and writes each selected .tch payload to <output>/<game_id>.tch so the
    user can inspect what the parser saw."""
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)

    output = tmp_path / "out"
    target = sorted(p.stem for p in _SAMPLES.glob("*.tch"))[0]
    rc = main([
        "--archive", str(archive),
        "--output", str(output),
        "--game-id", target,
    ])
    assert rc == 0

    # The targeted .tch is dumped into the output directory.
    dumped = output / f"{target}.tch"
    assert dumped.exists()
    # Content matches what the archive holds (modulo line endings — match
    # iter_archive's raw decode).
    sample_path = _SAMPLES / f"{target}.tch"
    assert dumped.read_bytes().decode("utf-8") == sample_path.read_bytes().decode("utf-8")

    # Only the targeted game contributes to parquet.
    for shard in output.glob("*.parquet"):
        table = pq.read_table(shard)
        ids = set(table.column("game_id").to_pylist())
        ids.discard("")
        assert ids <= {target}


def test_subset_does_not_decompress_discarded_entries(tmp_path, monkeypatch):
    """Regression guard. The pre-fix CLI did `sorted(iter_archive(...))[:N]`,
    which decompressed every payload in the archive before slicing — ~50 GB
    of RAM on the full corpus. Pin the cheap path: with `--subset N`,
    decompression must run at most N times, not once per archive entry."""
    archive = tmp_path / "sample.zst"
    _build_archive(_SAMPLES, archive)

    decompress_count = 0
    real_decompress = zstd.ZstdDecompressor.decompress

    def counting_decompress(self, *args, **kwargs):
        nonlocal decompress_count
        decompress_count += 1
        return real_decompress(self, *args, **kwargs)

    monkeypatch.setattr(zstd.ZstdDecompressor, "decompress", counting_decompress)

    output = tmp_path / "out"
    rc = main([
        "--archive", str(archive),
        "--output", str(output),
        "--subset", "1",
    ])
    assert rc == 0
    assert decompress_count <= 1, (
        f"--subset 1 should decompress at most 1 payload, but decompressed "
        f"{decompress_count}. The CLI is materialising the full archive."
    )


def test_only_matching_rounds_contribute_records_and_failed_games_logged(tmp_path):
    """ADR-0009 per-round granularity: rounds whose engine Ergebnis matches
    BSW's contribute records; the parent game is still logged to
    `known_bad_games.txt` if any of its rounds failed."""
    from tichu_training.bsw.parser import parse_tch
    from tichu_training.bsw.validate import validate_game

    output = tmp_path / "out"
    rc = main(["--input", str(_SAMPLES), "--output", str(output)])
    assert rc == 0

    # Ground truth: which rounds match per game?
    matching_by_game: dict[str, set[int]] = {}
    for path in sorted(_SAMPLES.glob("*.tch")):
        game = parse_tch(path.read_bytes().decode("utf-8"), game_id=path.stem)
        v = validate_game(game)
        matching_by_game[path.stem] = {r.round_index for r in v.rounds if r.matches}

    # Both samples have at least one failing round → both in known_bad_games.txt.
    known_bad = (output / "known_bad_games.txt").read_text().split()
    assert set(known_bad) == set(matching_by_game.keys())

    # Every parquet shard's (game_id, round_id) pairs must be a subset of the
    # matching set for the parent game.
    for shard in output.glob("*.parquet"):
        table = pq.read_table(shard)
        if table.num_rows == 0:
            continue
        for game_id, round_id in zip(
            table.column("game_id").to_pylist(),
            table.column("round_id").to_pylist(),
        ):
            assert round_id in matching_by_game[game_id], (
                f"{shard.name}: round {round_id} of game {game_id} "
                "should have been excluded (Ergebnis mismatch)"
            )
