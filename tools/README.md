# tichu-downloader

Tools for downloading, compressing, and decompressing Tichu game logs from
[brettspielwelt.de](http://tichulog.brettspielwelt.de).

All scripts use [uv](https://github.com/astral-sh/uv) for dependency management and can be
run directly without a virtualenv.

---

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

---

## 1. Download — `download.py`

Downloads `.tch` game log files from the BSW server.

```bash
./download.py
```

Files are written to the path configured in `OUTPUT_DIR` (default:
`/Users/andriesinze/workbench/tch-files`). Any IDs that could not be fetched
after retries are written to `failed.log`.

**Key settings** (edit at the top of the script):

| Variable | Default | Description |
|----------|---------|-------------|
| `OUTPUT_DIR` | `tch-files/` | Where `.tch` files are saved |
| `START` / `END` | `1` / `2417589` | ID range to download |
| `WORKERS` | `10` | Concurrent HTTP connections |
| `RETRIES` | `3` | Attempts per file before giving up |

Already-downloaded files are skipped automatically, so the script is safe to
re-run after an interrupted session.

---

## 2. Compress — `compress.py`

Packs every file in a folder into a single binary archive, compressed
individually with a shared [zstd](https://github.com/facebook/zstd) dictionary.
A JSON sidecar index (`.idx`) records each file's byte offset and length so
individual files can be extracted without scanning the whole archive.

```bash
./compress.py <input_dir> <output_archive>
```

**Example:**

```bash
./compress.py ~/workbench/tch-files ~/workbench/tch-files.zst
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--level` | `19` | Zstd compression level (1–22) |
| `--samples` | `1000` | Files sampled for dictionary training |
| `--dict-size` | `110000` | Dictionary size in bytes |
| `--seed` | `0` | RNG seed for reproducible sampling |

**Output files:**

- `<output_archive>` — raw concatenated compressed blobs
- `<output_archive>.idx` — JSON index with the dictionary and per-file offsets

The output directory is created automatically if it does not exist.

**Progress output:**

```
Scanning /path/to/tch-files … found 2,417,589 files (4.83 GB)
reading samples: 100%|██████████| 1000/1000 [00:01<00:00]
⠼ training dictionary… 7s
✓ dictionary trained
compressing: 42%|████████░░  | 1013k/2417k [04:12<05:51, 4012file/s, ratio=31.4%, out=284.7MB, file=1013421.tch]
```

---

## 3. Decompress — `decompress.py`

Extracts files from an archive produced by `compress.py`. Seeks directly to
each file's offset, so extracting a small subset is fast even for multi-million
file archives.

```bash
./decompress.py <archive> <output_dir>
```

**Example — extract everything:**

```bash
./decompress.py ~/workbench/tch-files.zst ~/workbench/tch-restored/
```

**Example — extract specific files:**

```bash
./decompress.py ~/workbench/tch-files.zst ~/workbench/tch-restored/ \
    --files 1.tch 42.tch 1337.tch
```

**Optional flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--files` | *(all)* | Space-separated list of relative paths to extract |
| `--index` | `<archive>.idx` | Path to the sidecar index if not co-located with the archive |

The output directory is created automatically if it does not exist.
