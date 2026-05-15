"""`compute_trueskill` CLI (slice 5a).

Reads `.tch` files (directory or `.tar.zst` archive) in chronological order
of game_id, runs a per-game team TrueSkill update, and writes a Parquet table
with one row per player_handle: mu, sigma, n_games.
"""

import argparse
import logging
import sys
from pathlib import Path

from tichu_training.bsw.parser import parse_tch
from tichu_training.ratings.calls import compute_call_stats
from tichu_training.ratings.post import apply_min_games, assign_skill_deciles
from tichu_training.ratings.source import iter_tch_sources
from tichu_training.ratings.stats import spearman_rho
from tichu_training.ratings.sweep import compute_ratings
from tichu_training.ratings.writer import write_ratings_parquet


_RHO_MIN_TICHU_CALLS = 5


log = logging.getLogger("compute_trueskill")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute TrueSkill ratings for BSW players.")
    p.add_argument("--input", required=True, metavar="PATH",
                   help="Directory of .tch files or a .tar.zst archive")
    p.add_argument("--output", required=True, metavar="FILE",
                   help="Output ratings Parquet file")
    p.add_argument("--min-games", type=int, default=20, metavar="N",
                   help="Minimum games required to appear in the output (default: 20)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("input path %s does not exist", input_path)
        return 2

    parsed_games = []
    parse_failures = 0
    for game_id, text in iter_tch_sources(input_path):
        try:
            parsed_games.append(parse_tch(text, game_id=str(game_id)))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping game %s: parse failed: %s", game_id, exc)
            parse_failures += 1
    log.info("parsed %d games successfully (%d skipped)", len(parsed_games), parse_failures)

    ratings = compute_ratings(parsed_games)
    log.info("rated %d distinct player handles", len(ratings))

    surviving = apply_min_games(ratings, min_games=args.min_games)
    log.info(
        "after --min-games=%d: %d/%d players retained",
        args.min_games, len(surviving), len(ratings),
    )

    deciles = assign_skill_deciles(surviving)
    call_stats = compute_call_stats(parsed_games)
    write_ratings_parquet(
        surviving,
        Path(args.output),
        deciles=deciles,
        call_stats=call_stats,
    )
    log.info("wrote ratings table to %s", args.output)

    _log_decile_share(deciles)
    _log_spearman(surviving, call_stats)
    return 0


def _log_decile_share(deciles: dict[str, int]) -> None:
    total = len(deciles)
    if total == 0:
        log.info("decile share (0..9): []  (no rated players)")
        return
    counts = [0] * 10
    for d in deciles.values():
        counts[d] += 1
    shares = [round(c / total, 3) for c in counts]
    log.info("decile share (0..9): %s", shares)


def _log_spearman(surviving, call_stats) -> None:
    mus: list[float] = []
    rates: list[float] = []
    for handle, rating in surviving.items():
        cs = call_stats.get(handle)
        if cs is None or cs.tichu_calls < _RHO_MIN_TICHU_CALLS:
            continue
        mus.append(rating.mu)
        rates.append(cs.tichu_wins / cs.tichu_calls)
    if len(mus) < _RHO_MIN_TICHU_CALLS:
        log.warning(
            "only %d players have tichu_calls >= %d; skipping spearman rho check",
            len(mus), _RHO_MIN_TICHU_CALLS,
        )
        return
    rho = spearman_rho(mus, rates)
    if rho is None:
        log.warning("spearman rho undefined (zero variance); skipping check")
        return
    log.info(
        "spearman rho(mu, tichu_success_rate) = %.3f over %d players (tichu_calls >= %d)",
        rho, len(mus), _RHO_MIN_TICHU_CALLS,
    )


if __name__ == "__main__":
    sys.exit(main())
