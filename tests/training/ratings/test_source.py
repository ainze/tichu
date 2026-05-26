"""Source iterator: yields (game_id, text) sorted ascending."""

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest
import zstandard as zstd

from tichu_training.ratings.source import iter_tch_sources


def _build_indexed_archive(items: list[tuple[str, str]], archive_path: Path) -> None:
    """Build a minimal ADR-0009 indexed-blob archive: concatenated dict-
    compressed payloads + a JSON sidecar at `<archive>.idx`. Mirrors
    `tools/compress.py`, which sorts entries by `int(stem)` so archive
    offset order equals chronological game_id order."""
    items = sorted(items, key=lambda it: int(it[0]) if it[0].isdigit() else it[0])
    payloads = [text.encode("utf-8") for _, text in items]
    dict_data = zstd.train_dictionary(8192, payloads * 50)
    cctx = zstd.ZstdCompressor(dict_data=dict_data, level=3)
    index: dict[str, list[int]] = {}
    offset = 0
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as out:
        for (stem, _), payload in zip(items, payloads):
            blob = cctx.compress(payload)
            out.write(blob)
            index[f"{stem}.tch"] = [offset, len(blob)]
            offset += len(blob)
    sidecar = {
        "version": 1,
        "level": 3,
        "dict": base64.b64encode(dict_data.as_bytes()).decode("ascii"),
        "files": index,
    }
    idx_path = archive_path.with_suffix(archive_path.suffix + ".idx")
    idx_path.write_text(json.dumps(sidecar))


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_directory_yields_sorted_by_int_stem(tmp_path):
    _write(tmp_path / "10.tch", "ten")
    _write(tmp_path / "2.tch", "two")
    _write(tmp_path / "100.tch", "hundred")
    out = list(iter_tch_sources(tmp_path))
    assert out == [(2, "two"), (10, "ten"), (100, "hundred")]


def test_directory_ignores_non_tch_files(tmp_path):
    _write(tmp_path / "1.tch", "one")
    _write(tmp_path / "known_bad_games.txt", "ignored")
    _write(tmp_path / "notes.md", "ignored")
    out = list(iter_tch_sources(tmp_path))
    assert out == [(1, "one")]


def test_tar_zst_archive_yields_sorted(tmp_path):
    archive = tmp_path / "games.tar.zst"
    # Build a tarball in memory, then zstd-compress to disk.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, text in [("10.tch", "ten"), ("2.tch", "two"), ("100.tch", "hundred")]:
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    cctx = zstd.ZstdCompressor()
    archive.write_bytes(cctx.compress(buf.getvalue()))

    out = list(iter_tch_sources(archive))
    assert out == [(2, "two"), (10, "ten"), (100, "hundred")]


def test_unknown_input_raises(tmp_path):
    missing = tmp_path / "nope.bin"
    missing.write_bytes(b"x")
    with pytest.raises(ValueError):
        list(iter_tch_sources(missing))


def test_zst_indexed_archive_yields_sorted(tmp_path):
    """Regression: the ADR-0009 indexed-blob archive (used by parse_bsw) must
    be readable by compute_trueskill too. Pre-fix this raised ValueError."""
    archive = tmp_path / "games.zst"
    _build_indexed_archive(
        [("10", "ten"), ("2", "two"), ("100", "hundred")], archive,
    )
    out = list(iter_tch_sources(archive))
    assert out == [(2, "two"), (10, "ten"), (100, "hundred")]


def test_zst_indexed_archive_subset_matches_parse_bsw_semantics(tmp_path):
    """--subset N takes the first N stems by ascending string-sort, matching
    parse_bsw exactly so a smoke ratings table aligns with a smoke parquet
    manifest."""
    archive = tmp_path / "games.zst"
    # Mixed-width stems so string-sort and int-sort differ for the picked set:
    # string-sort first-2 of ["10","100","2"] is ["10","100"]; int-sort would
    # pick ["2","10"]. Pin to parse_bsw's string-sort.
    _build_indexed_archive(
        [("10", "ten"), ("2", "two"), ("100", "hundred")], archive,
    )
    out = list(iter_tch_sources(archive, subset=2))
    # Subset picks {"10","100"} (string-sort first 2); yielded in chronological
    # int order so 10 comes before 100.
    assert out == [(10, "ten"), (100, "hundred")]


def test_zst_indexed_archive_game_ids_filter(tmp_path):
    archive = tmp_path / "games.zst"
    _build_indexed_archive(
        [("10", "ten"), ("2", "two"), ("100", "hundred")], archive,
    )
    out = list(iter_tch_sources(archive, game_ids={"2", "100"}))
    assert out == [(2, "two"), (100, "hundred")]


def test_zst_without_sidecar_raises_explanatory_error(tmp_path):
    """A bare `.zst` (no .idx) is rejected with a message that points at the
    missing sidecar — not a generic 'unsupported input'."""
    bare = tmp_path / "loose.zst"
    bare.write_bytes(b"\x28\xb5\x2f\xfd")  # zstd magic, enough to look real
    with pytest.raises(ValueError, match="sidecar"):
        list(iter_tch_sources(bare))


def test_directory_subset(tmp_path):
    _write(tmp_path / "10.tch", "ten")
    _write(tmp_path / "2.tch", "two")
    _write(tmp_path / "100.tch", "hundred")
    out = list(iter_tch_sources(tmp_path, subset=2))
    # Sort first by int, then slice: {2,10} keep, 100 drops.
    assert out == [(2, "two"), (10, "ten")]
