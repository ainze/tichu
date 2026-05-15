"""Yield BSW game sources in chronological order.

Two inputs are supported:
  - a directory of `.tch` files
  - a `.tar.zst` archive whose entries are `<game_id>.tch`

Both yield `(game_id_int, text)` tuples sorted by ascending `game_id_int`.
Non-`.tch` entries are skipped silently.
"""

import io
import tarfile
from pathlib import Path
from typing import Iterator

import zstandard as zstd


def iter_tch_sources(input_path: Path) -> Iterator[tuple[int, str]]:
    input_path = Path(input_path)
    if input_path.is_dir():
        yield from _iter_directory(input_path)
        return
    name = input_path.name.lower()
    if name.endswith(".tar.zst") or name.endswith(".tzst"):
        yield from _iter_tar_zst(input_path)
        return
    raise ValueError(f"unsupported input: {input_path} (expected directory or .tar.zst)")


def _iter_directory(directory: Path) -> Iterator[tuple[int, str]]:
    entries: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        if not p.is_file() or p.suffix != ".tch":
            continue
        try:
            game_id = int(p.stem)
        except ValueError:
            continue
        entries.append((game_id, p))
    entries.sort(key=lambda e: e[0])
    for game_id, path in entries:
        yield game_id, path.read_text(encoding="utf-8")


def _iter_tar_zst(archive: Path) -> Iterator[tuple[int, str]]:
    dctx = zstd.ZstdDecompressor()
    with archive.open("rb") as fh:
        decompressed = dctx.stream_reader(fh)
        # Buffer the full tar in memory; for a 2.4M-file archive we expect
        # the caller to stream once per CLI run. If memory becomes an issue
        # later we can switch to a streaming-tar reader.
        raw = decompressed.read()
    buf = io.BytesIO(raw)
    entries: list[tuple[int, bytes]] = []
    with tarfile.open(fileobj=buf, mode="r:") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".tch"):
                continue
            stem = Path(member.name).stem
            try:
                game_id = int(stem)
            except ValueError:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            entries.append((game_id, f.read()))
    entries.sort(key=lambda e: e[0])
    for game_id, data in entries:
        yield game_id, data.decode("utf-8")
