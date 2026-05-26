"""Yield BSW game sources in chronological order.

Three inputs are supported:
  - a directory of `.tch` files
  - a `.tar.zst` archive whose entries are `<game_id>.tch`
  - a `.zst` indexed-blob archive (with `<name>.zst.idx` sidecar) produced by
    `tools/compress.py` per [ADR-0009](../../../docs/adr/0009-bsw-ingest-streaming-pipeline.md)

All yield `(game_id_int, text)` tuples sorted by ascending `game_id_int`.
Non-`.tch` entries are skipped silently.

The `subset` / `game_ids` kwargs mirror `parse_bsw`'s flags so a smoke-run
ratings table is computed over the same games the parquet manifest captured.
For the indexed-blob path, `--subset N` semantics match parse_bsw exactly:
the first N stems by ascending string-sort of the sidecar's name set.
"""

import io
import tarfile
from pathlib import Path
from typing import Iterator

import zstandard as zstd

from tichu_training.bsw.archive import iter_archive, list_game_ids


def iter_tch_sources(
    input_path: Path,
    *,
    subset: int | None = None,
    game_ids: set[str] | None = None,
) -> Iterator[tuple[int, str]]:
    input_path = Path(input_path)
    if input_path.is_dir():
        yield from _iter_directory(input_path, subset=subset, game_ids=game_ids)
        return
    name = input_path.name.lower()
    if name.endswith(".tar.zst") or name.endswith(".tzst"):
        yield from _iter_tar_zst(input_path, subset=subset, game_ids=game_ids)
        return
    if name.endswith(".zst"):
        idx_path = input_path.with_suffix(input_path.suffix + ".idx")
        if idx_path.is_file():
            yield from _iter_zst_indexed(
                input_path, subset=subset, game_ids=game_ids,
            )
            return
        raise ValueError(
            f"{input_path}: .zst archive without sidecar {idx_path.name} — "
            "expected an indexed-blob archive per ADR-0009 "
            "(or rename to .tar.zst if it is a tarball)"
        )
    raise ValueError(
        f"unsupported input: {input_path} "
        "(expected directory, .tar.zst, or .zst with .zst.idx sidecar)"
    )


def _iter_directory(
    directory: Path,
    *,
    subset: int | None,
    game_ids: set[str] | None,
) -> Iterator[tuple[int, str]]:
    entries: list[tuple[int, Path]] = []
    for p in directory.iterdir():
        if not p.is_file() or p.suffix != ".tch":
            continue
        if game_ids is not None and p.stem not in game_ids:
            continue
        try:
            game_id = int(p.stem)
        except ValueError:
            continue
        entries.append((game_id, p))
    entries.sort(key=lambda e: e[0])
    if game_ids is None and subset is not None:
        entries = entries[:subset]
    for game_id, path in entries:
        yield game_id, path.read_text(encoding="utf-8")


def _iter_tar_zst(
    archive: Path,
    *,
    subset: int | None,
    game_ids: set[str] | None,
) -> Iterator[tuple[int, str]]:
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
            if game_ids is not None and stem not in game_ids:
                continue
            try:
                game_id = int(stem)
            except ValueError:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            entries.append((game_id, f.read()))
    entries.sort(key=lambda e: e[0])
    if game_ids is None and subset is not None:
        entries = entries[:subset]
    for game_id, data in entries:
        yield game_id, data.decode("utf-8")


def _iter_zst_indexed(
    archive: Path,
    *,
    subset: int | None,
    game_ids: set[str] | None,
) -> Iterator[tuple[int, str]]:
    """Stream an ADR-0009 indexed-blob archive in chronological game_id order.

    `iter_archive` yields in archive offset order; `tools/compress.py`
    writes entries sorted by `int(stem)` at build time, so offset order
    equals chronological game_id order by construction. We therefore
    stream straight through without materialising — keeps memory at one
    decompressed `.tch` payload in flight regardless of corpus size.

    `--subset N` matches parse_bsw's semantics: take the first N stems by
    string-sort of the sidecar's name set, so a 100k smoke ratings table
    covers byte-identical games to a 100k parquet manifest. Names are
    picked from the cheap sidecar listing before any zstd work, so
    discarded entries are never decompressed.
    """
    if game_ids is not None:
        keepers: set[str] | None = game_ids
    elif subset is not None:
        keepers = set(sorted(list_game_ids(archive))[:subset])
    else:
        keepers = None
    for stem, text in iter_archive(archive, game_ids=keepers):
        try:
            gid = int(stem)
        except ValueError:
            continue
        yield gid, text
