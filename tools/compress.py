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
import os
import random
import sys
import threading
import time
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm

DEFAULT_DICT_SIZE = 110_000
DEFAULT_SAMPLES = 1000
DEFAULT_LEVEL = 19


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress every file in a folder independently with a shared zstd dictionary."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing the files to compress.")
    parser.add_argument("output", type=Path, help="Output archive path. A sidecar .idx is written next to it.")
    parser.add_argument("--dict-size", type=int, default=DEFAULT_DICT_SIZE)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of files to sample for dictionary training.")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL, help="Zstd compression level (1-22).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling.")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.input_dir}")

    print(f"Scanning {args.input_dir} …", end=" ", flush=True)
    entries = sorted(
        (
            (Path(e.path), e.stat().st_size)
            for e in os.scandir(args.input_dir)
            if e.is_file()
        ),
        key=lambda x: int(x[0].stem) if x[0].stem.isdigit() else x[0].name,
    )
    if not entries:
        raise SystemExit(f"No files found in {args.input_dir}")
    files, sizes = zip(*entries)
    files = list(files)
    total_raw = sum(sizes)
    print(f"found {len(files):,} files ({total_raw / 1e9:.2f} GB)")

    rng = random.Random(args.seed)
    sample_files = rng.sample(files, min(args.samples, len(files)))
    samples = [p.read_bytes() for p in tqdm(sample_files, desc="reading samples", unit="file")]

    stop = threading.Event()
    def _spin() -> None:
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        t0 = time.monotonic()
        while not stop.is_set():
            elapsed = time.monotonic() - t0
            sys.stderr.write(f"\r{frames[i % len(frames)]} training dictionary… {elapsed:.0f}s")
            sys.stderr.flush()
            i += 1
            time.sleep(0.1)
        sys.stderr.write("\r✓ dictionary trained                    \n")
        sys.stderr.flush()

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    dict_data = zstd.train_dictionary(args.dict_size, samples)
    stop.set()
    spinner.join()

    cctx = zstd.ZstdCompressor(dict_data=dict_data, level=args.level)

    index: dict[str, list[int]] = {}
    offset = 0
    raw_total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        bar = tqdm(files, desc="compressing", unit="file", dynamic_ncols=True)
        for p in bar:
            data = p.read_bytes()
            blob = cctx.compress(data)
            out.write(blob)
            rel = p.relative_to(args.input_dir).as_posix()
            index[rel] = [offset, len(blob)]
            offset += len(blob)
            raw_total += len(data)
            ratio = offset / raw_total if raw_total else 0.0
            bar.set_postfix(ratio=f"{ratio:.1%}", out=f"{offset/1e6:.1f}MB", file=p.name)

    idx_path = args.output.with_suffix(args.output.suffix + ".idx")
    sidecar = {
        "version": 1,
        "level": args.level,
        "dict": base64.b64encode(dict_data.as_bytes()).decode("ascii"),
        "files": index,
    }
    idx_path.write_text(json.dumps(sidecar))

    ratio = offset / raw_total if raw_total else 0.0
    print(
        f"Wrote {len(files)} files: {raw_total:,} -> {offset:,} bytes "
        f"({ratio:.1%}). Archive: {args.output}, index: {idx_path}"
    )


if __name__ == "__main__":
    main()
