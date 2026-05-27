"""`compute_trueskill` CLI (slice 5a).

Reads `.tch` files (directory or `.tar.zst` archive) in chronological order
of game_id, runs a per-game team TrueSkill update, and writes a Parquet table
with one row per player_handle: mu, sigma, n_games.
"""

import argparse
import logging
import sys
from pathlib import Path

import trueskill
from tqdm import tqdm

from tichu_training.bsw.archive import count_entries
from tichu_training.bsw.parser import parse_tch
from tichu_training.ratings import calls as ratings_calls
from tichu_training.ratings import sweep as ratings_sweep
from tichu_training.ratings.calls import CallStats
from tichu_training.ratings.post import apply_min_games, assign_skill_deciles
from tichu_training.ratings.source import iter_tch_sources
from tichu_training.ratings.stats import spearman_rho
from tichu_training.ratings.sweep import PlayerRating
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
    p.add_argument("--subset", type=int, default=None, metavar="N",
                   help="Limit to first N games by sorted game_id (matches parse_bsw --subset)")
    p.add_argument("--game-id", action="append", default=None, metavar="ID", dest="game_ids",
                   help="Process only this game_id (repeatable). Useful for spot-checking.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("input path %s does not exist", input_path)
        return 2

    targeted_ids = set(args.game_ids) if args.game_ids else None
    total = _count_games(input_path, subset=args.subset, game_ids=targeted_ids)

    # Streaming pass: parse + TrueSkill update + call-stats update per game,
    # then discard. Keeps peak RAM at one ParsedGame in flight + the
    # accumulator dicts (sized by unique handles), instead of the previous
    # materialise-then-three-pass shape that needed many GB at 100k.
    env = trueskill.TrueSkill()
    ratings: dict[str, PlayerRating] = {}
    call_stats: dict[str, CallStats] = {}
    parsed_ok = 0
    parse_failures = 0
    bar = tqdm(total=total, unit="game", dynamic_ncols=True, desc="sweep")
    try:
        for game_id, text in iter_tch_sources(
            input_path, subset=args.subset, game_ids=targeted_ids,
        ):
            try:
                game = parse_tch(text, game_id=str(game_id))
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping game %s: parse failed: %s", game_id, exc)
                parse_failures += 1
                bar.update(1)
                bar.set_postfix(
                    ok=parsed_ok, failed=parse_failures, handles=len(ratings),
                    refresh=False,
                )
                continue
            ratings_sweep.update_for_game(game, ratings, env)
            ratings_calls.update_for_game(game, call_stats)
            parsed_ok += 1
            bar.update(1)
            bar.set_postfix(
                ok=parsed_ok, failed=parse_failures, handles=len(ratings),
                refresh=False,
            )
    finally:
        bar.close()
    log.info("parsed %d games successfully (%d skipped)", parsed_ok, parse_failures)
    log.info("rated %d distinct player handles", len(ratings))

    surviving = apply_min_games(ratings, min_games=args.min_games)
    log.info(
        "after --min-games=%d: %d/%d players retained",
        args.min_games, len(surviving), len(ratings),
    )

    deciles = assign_skill_deciles(surviving)
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


def _count_games(
    input_path: Path,
    *,
    subset: int | None,
    game_ids: set[str] | None,
) -> int | None:
    """Cheap upper-bound game count for the progress bar.

    Returns None when the count is not obtainable without scanning (i.e. for
    `.tar.zst` archives, which would require decompression). In that case
    tqdm displays a count-up rather than a percentage bar.
    """
    if game_ids is not None:
        return len(game_ids)
    if input_path.is_dir():
        n = sum(1 for _ in input_path.glob("*.tch"))
        return min(n, subset) if subset is not None else n
    name = input_path.name.lower()
    if name.endswith(".zst") and not (name.endswith(".tar.zst") or name.endswith(".tzst")):
        try:
            n = count_entries(input_path)
        except FileNotFoundError:
            return None
        return min(n, subset) if subset is not None else n
    return None


if __name__ == "__main__":
    sys.exit(main())
