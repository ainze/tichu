"""Source iterator: yields (game_id, text) sorted ascending."""

import io
import tarfile
from pathlib import Path

import pytest
import zstandard as zstd

from tichu_training.ratings.source import iter_tch_sources


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
