"""Emit per-decision training records as Parquet, sharded by `decision_type`.

Streaming pipeline (see [ADR-0009](../../../docs/adr/0009-bsw-ingest-streaming-pipeline.md)):
games come in as an iterator, each Round is replayed exactly once, only Rounds
whose engine-computed scores match BSW's recorded `Ergebnis` contribute
training records. Records are buffered per decision type and flushed to
ParquetWriter row-groups; one file per decision type is emitted regardless of
whether any record reached it. A game with at least one failing round is
reported via `StreamStats.failed_game_ids` for `known_bad_games.txt`.

Output layout: one shard per decision type, deterministically named
`{decision_type}_00000.parquet` so training jobs can glob shards without a
manifest. (Multi-shard splits can fill in higher suffixes later.)

Each row corresponds to one decision a BSW player made during a validated
round. The columns follow the v1 PRD schema:

  decision_type, game_id, round_id, timestamp, player_handle, action_taken,
  state, legal_actions_mask, round_outcome, round_won,
  featurizer_version, action_space_version, skill_decile, sample_weight

`state` and `legal_actions_mask` are reserved (filled at training-loop load
time, not at parse time). `featurizer_version` / `action_space_version` are
stamped from the live module constants. `skill_decile` is joined from a
ratings table if one is provided. `sample_weight` is `1.0` for games whose
`game_id >= recency_cutoff_game_id` (default `1855844`, first game of 2015),
else `recency_weight` (default `0.5`).
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_engine.cards import Card, SpecialCard
from tichu_engine.combinations import CardOrSpecial
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bsw.records import ParsedAction, ParsedGame, ParsedRound
from tichu_training.bsw.replay import ReplayResult, replay_round
from tichu_training.featurizer import FEATURIZER_VERSION


_DECISION_TYPE_BY_KIND: dict[str, str] = {
    "play": "play",
    "pass": "play",
    "schupfen": "schupfen",
    "tichu": "call_tichu",
    "grand_tichu": "call_grand_tichu",
    "wish": "wish",
    "dragon_give": "dragon_assignment",
}


_KNOWN_DECISION_TYPES = ("play", "schupfen", "call_tichu", "call_grand_tichu", "wish", "dragon_assignment")

_DEFAULT_RECENCY_CUTOFF: int = 1855844
_DEFAULT_RECENCY_WEIGHT: float = 0.5

_DEFAULT_ROWS_PER_FLUSH: int = 50_000


_SCHEMA = pa.schema([
    ("decision_type", pa.string()),
    ("game_id", pa.string()),
    ("round_id", pa.int32()),
    ("timestamp", pa.int64()),
    ("player_handle", pa.string()),
    ("action_taken", pa.string()),
    ("state", pa.binary()),
    ("legal_actions_mask", pa.binary()),
    ("round_outcome", pa.int32()),
    ("round_won", pa.bool_()),
    ("featurizer_version", pa.string()),
    ("action_space_version", pa.string()),
    ("skill_decile", pa.int32()),
    ("sample_weight", pa.float32()),
])


@dataclass
class _Record:
    decision_type: str
    game_id: str
    round_id: int
    timestamp: int | None
    player_handle: str
    action_taken: str
    state: bytes | None
    legal_actions_mask: bytes | None
    round_outcome: int
    round_won: bool
    featurizer_version: str | None
    action_space_version: str | None
    skill_decile: int | None
    sample_weight: float


@dataclass
class RoundFailure:
    """One round that failed replay validation. Surfaced via `StreamStats`
    and `failure_details.tsv` so engine-fix iteration can group failures by
    category and pick the highest-leverage targets."""
    game_id: str
    round_id: int
    mode: str   # "illegal_action" | "score_mismatch"
    detail: str


@dataclass
class _GameResult:
    """Per-game output of the replay+emit step. Picklable so it can cross
    a process boundary when stream_to_parquet runs with workers > 1."""
    game_id: str
    records: list["_Record"]
    rounds_total: int
    rounds_matched: int
    failures: list[RoundFailure]

    @property
    def had_failure(self) -> bool:
        return self.rounds_matched < self.rounds_total


def _process_game(
    game: ParsedGame,
    *,
    skill_lookup: dict[str, int | None],
    recency_cutoff_game_id: int,
    recency_weight: float,
) -> _GameResult:
    """Replay every round of `game` and emit per-decision records for the
    rounds whose engine Ergebnis matches BSW's. Module-level (picklable) so it
    runs in ProcessPoolExecutor workers."""
    sample_weight = _sample_weight_for(
        game.game_id or "", recency_cutoff_game_id, recency_weight,
    )
    records: list[_Record] = []
    failures: list[RoundFailure] = []
    rounds_total = 0
    rounds_matched = 0
    game_id = game.game_id or "<unknown>"
    for parsed_round in game.rounds:
        replay = replay_round(parsed_round)
        rounds_total += 1
        failure = _classify_round_failure(game_id, parsed_round, replay)
        if failure is not None:
            failures.append(failure)
            continue
        rounds_matched += 1
        records.extend(_emit_records_for_round(
            game, parsed_round, replay,
            skill_lookup=skill_lookup,
            sample_weight=sample_weight,
        ))
    return _GameResult(
        game_id=game_id,
        records=records,
        rounds_total=rounds_total,
        rounds_matched=rounds_matched,
        failures=failures,
    )


def _classify_round_failure(
    game_id: str, parsed: ParsedRound, replay: ReplayResult,
) -> RoundFailure | None:
    """Return a RoundFailure if the replay disagrees with BSW, else None."""
    if replay.final_state is None:
        kind = (replay.illegal_action.kind if replay.illegal_action else "unknown")
        reason = replay.illegal_reason or "no reason recorded"
        return RoundFailure(
            game_id=game_id,
            round_id=parsed.round_index,
            mode="illegal_action",
            detail=f"{kind}: {reason}",
        )
    actual = replay.final_state.public.scores
    if actual != parsed.ergebnis:
        return RoundFailure(
            game_id=game_id,
            round_id=parsed.round_index,
            mode="score_mismatch",
            detail=f"engine={actual} bsw={parsed.ergebnis}",
        )
    return None


@dataclass
class StreamStats:
    """Output of `stream_to_parquet` — counts and the games with at least
    one replay-failed round (for `known_bad_games.txt`)."""
    games_total: int = 0
    games_fully_matched: int = 0
    games_with_failed_rounds: int = 0
    rounds_total: int = 0
    rounds_matched: int = 0
    row_counts: dict[str, int] = field(default_factory=lambda: {t: 0 for t in _KNOWN_DECISION_TYPES})
    failed_game_ids: list[str] = field(default_factory=list)
    round_failures: list[RoundFailure] = field(default_factory=list)


def stream_to_parquet(
    games: Iterable[ParsedGame],
    output_dir: Path,
    *,
    ratings_path: str | Path | None = None,
    recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
    recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
    rows_per_flush: int = _DEFAULT_ROWS_PER_FLUSH,
    workers: int = 1,
    chunksize: int = 16,
    on_game_done: Callable[["StreamStats"], None] | None = None,
) -> StreamStats:
    """Stream parsed games to per-decision Parquet shards, filtering at the
    Round granularity (see ADR-0009).

    One `ParquetWriter` is opened lazily per decision type the first time a
    row of that type is buffered, and closed in `try/finally` so partial runs
    leave valid (footer-written) Parquet files. After the iterator drains,
    any decision types that never saw a record get an empty file written so
    the on-disk shape is invariant.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skill_lookup = _load_skill_lookup(ratings_path) if ratings_path else {}

    stats = StreamStats()
    buffers: dict[str, list[_Record]] = {t: [] for t in _KNOWN_DECISION_TYPES}
    writers: dict[str, pq.ParquetWriter] = {}

    def _flush(decision_type: str) -> None:
        records = buffers[decision_type]
        if not records:
            return
        table = _to_table(records)
        if decision_type not in writers:
            path = output_dir / f"{decision_type}_00000.parquet"
            writers[decision_type] = pq.ParquetWriter(path, _SCHEMA)
        writers[decision_type].write_table(table)
        records.clear()

    def _apply(result: _GameResult) -> None:
        stats.games_total += 1
        stats.rounds_total += result.rounds_total
        stats.rounds_matched += result.rounds_matched
        if result.had_failure:
            stats.games_with_failed_rounds += 1
            stats.failed_game_ids.append(result.game_id)
        else:
            stats.games_fully_matched += 1
        stats.round_failures.extend(result.failures)
        for record in result.records:
            buffers[record.decision_type].append(record)
            stats.row_counts[record.decision_type] += 1
        for decision_type, buf in buffers.items():
            if len(buf) >= rows_per_flush:
                _flush(decision_type)
        if on_game_done is not None:
            on_game_done(stats)

    worker = partial(
        _process_game,
        skill_lookup=skill_lookup,
        recency_cutoff_game_id=recency_cutoff_game_id,
        recency_weight=recency_weight,
    )

    try:
        if workers <= 1:
            for game in games:
                _apply(worker(game))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(worker, games, chunksize=chunksize):
                    _apply(result)
        for decision_type in _KNOWN_DECISION_TYPES:
            _flush(decision_type)
            if decision_type not in writers:
                path = output_dir / f"{decision_type}_00000.parquet"
                writers[decision_type] = pq.ParquetWriter(path, _SCHEMA)
    finally:
        for w in writers.values():
            w.close()
    return stats


def _emit_records_for_round(
    game: ParsedGame,
    parsed_round: ParsedRound,
    replay: ReplayResult,
    *,
    skill_lookup: dict[str, int | None],
    sample_weight: float,
):
    team_outcome = parsed_round.ergebnis[0] - parsed_round.ergebnis[1]
    for parsed_action, _engine_action in replay.decisions:
        decision_type = _DECISION_TYPE_BY_KIND.get(parsed_action.kind)
        if decision_type is None:
            continue
        player = parsed_action.player
        if 0 <= player < 4:
            handle = game.handles[player]
        else:
            handle = ""
        team = player % 2 if 0 <= player < 4 else 0
        round_won = (
            parsed_round.ergebnis[team] > parsed_round.ergebnis[1 - team]
        )
        yield _Record(
            decision_type=decision_type,
            game_id=game.game_id or "",
            round_id=parsed_round.round_index,
            timestamp=None,
            player_handle=handle,
            action_taken=_serialise_action(parsed_action),
            state=None,
            legal_actions_mask=None,
            round_outcome=team_outcome,
            round_won=round_won,
            featurizer_version=FEATURIZER_VERSION,
            action_space_version=ACTION_SPACE_VERSION,
            skill_decile=skill_lookup.get(handle),
            sample_weight=sample_weight,
        )


def _load_skill_lookup(ratings_path: str | Path) -> dict[str, int | None]:
    table = pq.read_table(ratings_path, columns=["player_handle", "skill_decile"])
    handles = table.column("player_handle").to_pylist()
    deciles = table.column("skill_decile").to_pylist()
    return {h: d for h, d in zip(handles, deciles)}


def _sample_weight_for(game_id: str, cutoff: int, low_weight: float) -> float:
    try:
        gid = int(game_id)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if gid >= cutoff else low_weight


def _serialise_action(action: ParsedAction) -> str:
    if action.kind in ("play", "schupfen"):
        if action.kind == "schupfen":
            return (
                f"schupfen:next={_card_str(action.schupfen_to_next)},"
                f"partner={_card_str(action.schupfen_to_partner)},"
                f"prev={_card_str(action.schupfen_to_previous)}"
            )
        return f"play:{','.join(_card_str(c) for c in action.cards)}"
    if action.kind == "pass":
        return "pass"
    if action.kind == "wish":
        return f"wish:{action.wish_rank}"
    if action.kind == "dragon_give":
        return f"dragon_give:{action.dragon_target}"
    if action.kind in ("tichu", "grand_tichu"):
        return action.kind
    return action.kind


def _card_str(card: CardOrSpecial | None) -> str:
    if card is None:
        return ""
    if isinstance(card, SpecialCard):
        return card.name
    if isinstance(card, Card):
        return f"{card.suit.value}-{card.rank}"
    return str(card)


def _to_table(records: list[_Record]) -> pa.Table:
    columns: dict[str, list] = {f.name: [] for f in _SCHEMA}
    for r in records:
        columns["decision_type"].append(r.decision_type)
        columns["game_id"].append(r.game_id)
        columns["round_id"].append(r.round_id)
        columns["timestamp"].append(r.timestamp)
        columns["player_handle"].append(r.player_handle)
        columns["action_taken"].append(r.action_taken)
        columns["state"].append(r.state)
        columns["legal_actions_mask"].append(r.legal_actions_mask)
        columns["round_outcome"].append(r.round_outcome)
        columns["round_won"].append(r.round_won)
        columns["featurizer_version"].append(r.featurizer_version)
        columns["action_space_version"].append(r.action_space_version)
        columns["skill_decile"].append(r.skill_decile)
        columns["sample_weight"].append(r.sample_weight)
    return pa.Table.from_pydict(columns, schema=_SCHEMA)
