"""`parse_bsw` CLI.

Streams `.tch` game logs into per-decision Parquet shards. Two ingest modes:

* `--input DIR` reads loose `.tch` files from a directory.
* `--archive FILE` streams `.tch` payloads out of a zstd archive produced by
  `tools/compress.py` (no scratch directory needed).

Each game is replayed exactly once through the engine; records are emitted at
Round granularity (see [ADR-0009](../../../docs/adr/0009-bsw-ingest-streaming-pipeline.md)).
Games with at least one failing round are recorded in `known_bad_games.txt`;
their matching rounds still contribute training rows.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from tichu_training.bsw.archive import count_entries, iter_archive, list_game_ids
from tichu_training.bsw.to_parquet import StreamStats, stream_raw_to_parquet


log = logging.getLogger("parse_bsw")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parse raw BSW log files into training Parquet shards.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", metavar="DIR", help="Directory containing BSW .tch log files")
    source.add_argument("--archive", metavar="FILE", help="Zstd archive (.zst) of .tch files; index sidecar must sit alongside")
    p.add_argument("--output", required=True, metavar="DIR", help="Output directory for Parquet shards")
    p.add_argument("--subset", type=int, default=None, metavar="N", help="Limit to first N games (sorted by game_id)")
    p.add_argument("--game-id", action="append", default=None, metavar="ID", dest="game_ids",
                   help="Process only this game_id (repeatable). Also dumps the raw .tch text to "
                        "<output>/<game_id>.tch for inspection. Useful for debugging parse/replay failures.")
    p.add_argument("--trueskill", metavar="FILE",
                   help="TrueSkill ratings Parquet to join `skill_decile` from (by player_handle)")
    p.add_argument("--recency-cutoff-game-id", type=int, default=1855844, metavar="N",
                   help="Games with game_id >= N get sample_weight 1.0; below get --recency-weight (default 1855844 = first 2015 game)")
    p.add_argument("--recency-weight", type=float, default=0.5, metavar="W",
                   help="Sample weight for pre-cutoff games (default 0.5)")
    p.add_argument("--workers", type=int, default=1, metavar="N",
                   help="Number of replay worker processes (default 1 = single-threaded). "
                        "Forced to 1 when --game-id is set so debug output remains in-process.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(args.output)
    targeted_ids: set[str] | None = set(args.game_ids) if args.game_ids else None

    if args.input is not None:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            log.error("input directory %s does not exist", input_dir)
            return 2
        source_pairs = _iter_dir(input_dir, subset=args.subset, game_ids=targeted_ids)
        total = _count_dir(input_dir, subset=args.subset, game_ids=targeted_ids)
    else:
        archive_path = Path(args.archive)
        if not archive_path.is_file():
            log.error("archive %s does not exist", archive_path)
            return 2
        source_pairs = _iter_archive_pairs(archive_path, subset=args.subset, game_ids=targeted_ids)
        total = _count_archive(archive_path, subset=args.subset, game_ids=targeted_ids)

    output_dir.mkdir(parents=True, exist_ok=True)

    if targeted_ids is not None:
        # Debug mode: dump the raw .tch text of each targeted game next to the
        # parquet output so the user can read what the parser was given.
        source_pairs = _dumping(source_pairs, output_dir)

    bar = tqdm(total=total, unit="game", dynamic_ncols=True)

    def _on_game_done(stats: StreamStats) -> None:
        bar.update(1)
        bar.set_postfix(
            valid=stats.games_fully_matched,
            failed=stats.games_with_failed_rounds + len(stats.parse_failures),
            rows=sum(stats.row_counts.values()),
            refresh=False,
        )

    # Debug mode (--game-id) keeps everything in-process so verbose logs and
    # the .tch dump live next to the parquet output.
    effective_workers = 1 if targeted_ids is not None else max(1, args.workers)

    try:
        stats = stream_raw_to_parquet(
            source_pairs,
            output_dir,
            ratings_path=args.trueskill,
            recency_cutoff_game_id=args.recency_cutoff_game_id,
            recency_weight=args.recency_weight,
            workers=effective_workers,
            on_game_done=_on_game_done,
        )
    finally:
        bar.close()

    log.info(
        "replay validation: %d/%d rounds matched (%.3f%%); %d games fully matched, %d games had at least one failing round, %d parse failures",
        stats.rounds_matched, stats.rounds_total,
        (stats.rounds_matched / stats.rounds_total * 100) if stats.rounds_total else 0.0,
        stats.games_fully_matched, stats.games_with_failed_rounds, len(stats.parse_failures),
    )

    known_bad_path = output_dir / "known_bad_games.txt"
    with known_bad_path.open("w", encoding="utf-8") as fh:
        for failed_id in stats.failed_game_ids:
            fh.write(f"{failed_id}\n")
    log.info("wrote known_bad_games.txt with %d entries", len(stats.failed_game_ids))

    parse_failures_path = output_dir / "parse_failures.txt"
    with parse_failures_path.open("w", encoding="utf-8") as fh:
        for parse_failed in stats.parse_failures:
            fh.write(f"{parse_failed}\n")
    log.info("wrote parse_failures.txt with %d entries", len(stats.parse_failures))

    details_path = output_dir / "failure_details.tsv"
    with details_path.open("w", encoding="utf-8") as fh:
        fh.write("game_id\tround_id\tmode\tdetail\n")
        for f in stats.round_failures:
            fh.write(f"{f.game_id}\t{f.round_id}\t{f.mode}\t{f.detail}\n")
    n_illegal = sum(1 for f in stats.round_failures if f.mode == "illegal_action")
    n_score = sum(1 for f in stats.round_failures if f.mode == "score_mismatch")
    log.info(
        "wrote failure_details.tsv with %d entries (%d illegal_action, %d score_mismatch)",
        len(stats.round_failures), n_illegal, n_score,
    )

    for decision_type, n in stats.row_counts.items():
        log.info("shard %s.parquet: %d rows", decision_type, n)

    return 0


def _iter_dir(
    input_dir: Path, *, subset: int | None, game_ids: set[str] | None,
) -> Iterator[tuple[str, str]]:
    paths = sorted(input_dir.glob("*.tch"))
    if game_ids is not None:
        paths = [p for p in paths if p.stem in game_ids]
    elif subset is not None:
        paths = paths[:subset]
    for path in paths:
        yield path.stem, path.read_text(encoding="utf-8")


def _count_dir(
    input_dir: Path, *, subset: int | None, game_ids: set[str] | None,
) -> int:
    if game_ids is not None:
        return sum(1 for p in input_dir.glob("*.tch") if p.stem in game_ids)
    n = len(list(input_dir.glob("*.tch")))
    return min(n, subset) if subset is not None else n


def _iter_archive_pairs(
    archive_path: Path, *, subset: int | None, game_ids: set[str] | None,
) -> Iterator[tuple[str, str]]:
    if game_ids is not None:
        yield from iter_archive(archive_path, game_ids=game_ids)
        return
    if subset is None:
        yield from iter_archive(archive_path)
        return
    # `--subset N` in dir mode is "first N by sorted game_id". Match that —
    # picking the names from the cheap sidecar listing so we never decompress
    # entries we'll discard. (A naive `sorted(iter_archive(...))[:N]` would
    # decompress every payload in the archive — ~50 GB on the full corpus.)
    keepers = set(sorted(list_game_ids(archive_path))[:subset])
    yield from iter_archive(archive_path, game_ids=keepers)


def _count_archive(
    archive_path: Path, *, subset: int | None, game_ids: set[str] | None,
) -> int:
    if game_ids is not None:
        return len(game_ids & set(list_game_ids(archive_path)))
    n = count_entries(archive_path)
    return min(n, subset) if subset is not None else n


def _dumping(
    pairs: Iterator[tuple[str, str]], output_dir: Path,
) -> Iterator[tuple[str, str]]:
    """Write each tch_text to <output_dir>/<game_id>.tch as it streams past."""
    for game_id, text in pairs:
        (output_dir / f"{game_id}.tch").write_bytes(text.encode("utf-8"))
        yield game_id, text


if __name__ == "__main__":
    sys.exit(main())
