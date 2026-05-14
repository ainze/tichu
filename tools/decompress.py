#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zstandard",
#   "tqdm",
# ]
# ///

import argparse
import base64
import json
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompress an archive produced by compress.py."
    )
    parser.add_argument("archive", type=Path, help="Archive file written by compress.py.")
    parser.add_argument("output_dir", type=Path, help="Folder to write decompressed files into.")
    parser.add_argument("--index", type=Path, default=None, help="Sidecar index (defaults to <archive>.idx).")
    parser.add_argument("--files", nargs="*", metavar="NAME", help="Only extract these relative paths (all if omitted).")
    args = parser.parse_args()

    idx_path = args.index or args.archive.with_suffix(args.archive.suffix + ".idx")
    if not idx_path.exists():
        raise SystemExit(f"Index not found: {idx_path}")
    if not args.archive.exists():
        raise SystemExit(f"Archive not found: {args.archive}")

    sidecar = json.loads(idx_path.read_text())
    dict_bytes = base64.b64decode(sidecar["dict"])
    dict_data = zstd.ZstdCompressionDict(dict_bytes)
    dctx = zstd.ZstdDecompressor(dict_data=dict_data)

    all_files: dict[str, list[int]] = sidecar["files"]
    targets = {k: v for k, v in all_files.items() if not args.files or k in args.files}
    if not targets:
        raise SystemExit("No matching files in index.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.archive.open("rb") as arc:
        for rel, (offset, length) in tqdm(targets.items(), desc="decompressing", unit="file"):
            arc.seek(offset)
            blob = arc.read(length)
            data = dctx.decompress(blob)
            dest = args.output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)

    print(f"Extracted {len(targets)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
