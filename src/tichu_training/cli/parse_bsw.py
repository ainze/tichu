"""`parse_bsw` CLI.

Reads `.tch` files from --input, replays each through the rules engine,
emits Parquet shards by decision_type to --output, and writes the IDs of any
games whose engine score does not match BSW's recorded `Ergebnis` to a
`known_bad_games.txt` in the output directory.
"""

import argparse
import logging
import sys
from pathlib import Path

from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.to_parquet import write_parquet_shards
from tichu_training.bsw.validate import validate_corpus


log = logging.getLogger("parse_bsw")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parse raw BSW log files into training Parquet shards.")
    p.add_argument("--input", required=True, metavar="DIR", help="Directory containing BSW .tch log files")
    p.add_argument("--output", required=True, metavar="DIR", help="Output directory for Parquet shards")
    p.add_argument("--subset", type=int, default=None, metavar="N", help="Limit to first N games")
    p.add_argument("--trueskill", metavar="FILE",
                   help="TrueSkill ratings Parquet to join `skill_decile` from (by player_handle)")
    p.add_argument("--recency-cutoff-game-id", type=int, default=1855844, metavar="N",
                   help="Games with game_id >= N get sample_weight 1.0; below get --recency-weight (default 1855844 = first 2015 game)")
    p.add_argument("--recency-weight", type=float, default=0.5, metavar="W",
                   help="Sample weight for pre-cutoff games (default 0.5)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        log.error("input directory %s does not exist", input_dir)
        return 2

    paths = sorted(input_dir.glob("*.tch"))
    if args.subset is not None:
        paths = paths[: args.subset]
    if not paths:
        log.error("no .tch files found under %s", input_dir)
        return 2
    log.info("found %d .tch files", len(paths))

    games = []
    parse_failures: list[str] = []
    for path in paths:
        game_id = path.stem
        try:
            game = parse_tch(path.read_text(encoding="utf-8"), game_id=game_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: parse failed: %s", path.name, exc)
            parse_failures.append(game_id)
            continue
        games.append(game)

    log.info("parsed %d games successfully (%d skipped)", len(games), len(parse_failures))

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = validate_corpus(games)
    log.info(
        "replay validation: %d/%d games match BSW Ergebnis (pass rate %.3f%%)",
        stats.games_matched, stats.games_total, stats.pass_rate * 100,
    )

    known_bad_path = output_dir / "known_bad_games.txt"
    with known_bad_path.open("w", encoding="utf-8") as fh:
        for failed_id in stats.failed_game_ids:
            fh.write(f"{failed_id}\n")
        for parse_failed in parse_failures:
            fh.write(f"{parse_failed}\n")
    log.info("wrote known_bad_games.txt with %d entries", len(stats.failed_game_ids) + len(parse_failures))

    counts = write_parquet_shards(
        games,
        output_dir,
        ratings_path=args.trueskill,
        recency_cutoff_game_id=args.recency_cutoff_game_id,
        recency_weight=args.recency_weight,
    )
    for decision_type, n in counts.items():
        log.info("shard %s.parquet: %d rows", decision_type, n)

    return 0


if __name__ == "__main__":
    sys.exit(main())
