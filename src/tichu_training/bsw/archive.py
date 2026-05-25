"""Stream `.tch` payloads out of a zstd archive produced by ``tools/compress.py``.

The archive is a concatenation of independently-compressed blobs, addressed by
a JSON sidecar (``<archive>.idx``) holding the shared dictionary and per-file
``(offset, length)`` pairs. ``iter_archive`` decodes each blob in memory and
yields ``(game_id, tch_text)``, never materialising the decompressed payloads
on disk.
"""

import base64
import json
from pathlib import Path
from typing import Iterator

import zstandard as zstd


def count_entries(archive_path: Path) -> int:
    """Return the number of `.tch` payloads addressable by *archive_path*'s
    sidecar index. Cheap — only reads the JSON sidecar."""
    return len(_load_sidecar(archive_path)["files"])


def list_game_ids(archive_path: Path) -> list[str]:
    """Return all game_ids in the archive, in offset order. Cheap — only reads
    the JSON sidecar; no zstd work, no payload decompression."""
    sidecar = _load_sidecar(archive_path)
    entries = sorted(sidecar["files"].items(), key=lambda kv: kv[1][0])
    return [Path(rel).stem for rel, _ in entries]


def iter_archive(
    archive_path: Path,
    *,
    game_ids: set[str] | None = None,
) -> Iterator[tuple[str, str]]:
    """Yield ``(game_id, tch_text)`` for every ``.tch`` file in *archive_path*.

    ``game_id`` is the filename stem (``"2417500"`` for ``2417500.tch``).
    Entries are yielded in archive offset order so reads are sequential.

    If *game_ids* is given, only entries whose stem is in that set are
    decompressed and yielded; non-matching entries are skipped without
    touching the archive file. This makes ``--subset`` cheap: callers can
    pick the names they want from :func:`list_game_ids` first and never
    decompress the discarded payloads.
    """
    archive_path = Path(archive_path)
    sidecar = _load_sidecar(archive_path)
    dict_data = zstd.ZstdCompressionDict(base64.b64decode(sidecar["dict"]))
    dctx = zstd.ZstdDecompressor(dict_data=dict_data)

    entries: list[tuple[str, int, int]] = [
        (rel, offset, length) for rel, (offset, length) in sidecar["files"].items()
    ]
    entries.sort(key=lambda e: e[1])

    with archive_path.open("rb") as arc:
        for rel, offset, length in entries:
            game_id = Path(rel).stem
            if game_ids is not None and game_id not in game_ids:
                continue
            arc.seek(offset)
            blob = arc.read(length)
            payload = dctx.decompress(blob)
            yield game_id, payload.decode("utf-8")


def _load_sidecar(archive_path: Path) -> dict:
    archive_path = Path(archive_path)
    idx_path = archive_path.with_suffix(archive_path.suffix + ".idx")
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    if not idx_path.is_file():
        raise FileNotFoundError(f"index not found: {idx_path}")
    return json.loads(idx_path.read_text())
