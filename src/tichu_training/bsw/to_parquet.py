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
round. The columns follow the v3 schema:

  decision_type, game_id, round_id, timestamp, player_handle, action_taken,
  state, legal_actions_mask, round_outcome, round_won, game_won,
  featurizer_version, action_space_version, skill_decile, sample_weight

`state` and `legal_actions_mask` are reserved (filled at training-loop load
time, not at parse time). `featurizer_version` / `action_space_version` are
stamped from the live module constants. `skill_decile` is joined from a
ratings table if one is provided. `sample_weight` is `1.0` for games whose
`game_id >= recency_cutoff_game_id` (default `1855844`, first game of 2015),
else `recency_weight` (default `0.5`). `round_won` and `game_won` are both
team-relative to the row's acting player; `game_won` is NULL for rows from
an Incomplete Session (see `bsw.records.game_team_totals`). The parquet
schema is versioned by directory name (`parquet_<scale>_v<N>`), not by a
per-row column — see ADR-0013.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_engine.cards import Card, SpecialCard
from tichu_engine.combinations import CardOrSpecial
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bc.dataset import BCExample, _KIND_TO_DECISION_TYPE
from tichu_training.bsw.parser import parse_tch
from tichu_training.bsw.records import ParsedAction, ParsedGame, ParsedRound, game_team_totals
from tichu_training.bsw.replay import ReplayResult, replay_round
from tichu_training.featurizer import FEATURIZER_VERSION


# Neutral skill bucket (the "11th row") for handles with no rating, matching
# `ParquetBCDataset`'s `_neutral_decile = skill_buckets`. The bundle's meta
# dtype stores skill_decile as a uint8, so it must always be a concrete int —
# unlike the parquet path's nullable column.
_DEFAULT_SKILL_BUCKETS: int = 10


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
    ("game_won", pa.bool_()),
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
    game_won: bool | None
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
    # Populated only when `emit_bundle=True`: the BCExamples for the same
    # validated rounds, featurised at each decision boundary. Empty otherwise
    # so the no-bundle path pays nothing. See the consolidation handoff —
    # these come from the *same* replay that produced `records`, so the
    # engine runs once per round instead of once per pipeline.
    bc_examples: list["BCExample"] = field(default_factory=list)
    # Populated when the corresponding task is in `bundle_tasks` — the
    # per-task examples for the same validated rounds, from the same replay
    # (ADR-0020). Calls split by type (tichu reuses the full replay; grand is
    # synthetic deal-time). Schupfen is synthetic pre-schupfen. Empty otherwise.
    call_tichu_examples: list = field(default_factory=list)
    call_grand_examples: list = field(default_factory=list)
    schupfen_examples: list = field(default_factory=list)
    belief_examples: list = field(default_factory=list)
    # True if the game's summed ergebnis reaches 1000 for at least one team
    # (Complete Game); False if it's an Incomplete Session. None when parse
    # failed and there's no ergebnis to read.
    is_complete_game: bool | None = None
    # Set by `_parse_and_process` when parse_tch raised. `records` / `rounds_*`
    # are empty in that case; `parse_error` is the exception's `str()`.
    parse_error: str | None = None

    @property
    def had_failure(self) -> bool:
        return self.rounds_matched < self.rounds_total


def _int_game_id(game_id: str) -> int:
    """Parse a BSW game_id to int for the bundle's `game_id u4` provenance
    column; 0 if not parseable (the bundle treats 0 as 'unknown')."""
    try:
        return int(game_id)
    except (TypeError, ValueError):
        return 0


def _stamp_provenance(examples: list, game_id_int: int, round_id: int, neutral: int):
    """Stamp (game_id, round_id) onto the per-task examples and coerce a None
    skill_decile to the Neutral Skill Decile — the bundle meta stores skill as
    a uint8, so it must be a concrete int (the parquet path's column is
    nullable; these bundles are not)."""
    for e in examples:
        e.game_id = game_id_int
        e.round_id = round_id
        if e.skill_decile is None:
            e.skill_decile = neutral
    return examples


def _process_game(
    game: ParsedGame,
    *,
    skill_lookup: dict[str, int | None],
    recency_cutoff_game_id: int,
    recency_weight: float,
    bundle_tasks: frozenset[str] = frozenset(),
    neutral_decile: int = _DEFAULT_SKILL_BUCKETS,
) -> _GameResult:
    """Replay every round of `game` and emit per-decision records for the
    rounds whose engine Ergebnis matches BSW's. Module-level (picklable) so it
    runs in ProcessPoolExecutor workers.

    For each task in `bundle_tasks`, the *same* replay also yields that task's
    featurised examples into the result — the consolidation that lets one
    engine pass feed the parquet manifest and every materialised bundle
    (ADR-0020). Tichu reuses the full replay's first-non-Pass-Play states;
    grand-tichu and schupfen build their synthetic states from the parsed
    round; BC featurises at every BC decision boundary."""
    from tichu_training.bc.call_emit import (
        grand_tichu_examples_for_round,
        tichu_examples_for_round,
    )
    from tichu_training.bc.schupfen_training import _schupfen_examples_for_round
    from tichu_training.belief.emit import belief_examples_for_round

    sample_weight = _sample_weight_for(
        game.game_id or "", recency_cutoff_game_id, recency_weight,
    )
    team_totals = game_team_totals(game)
    records: list[_Record] = []
    bc_examples: list[BCExample] = []
    call_tichu_examples: list = []
    call_grand_examples: list = []
    schupfen_examples: list = []
    belief_examples: list = []
    failures: list[RoundFailure] = []
    rounds_total = 0
    rounds_matched = 0
    game_id = game.game_id or "<unknown>"
    game_id_int = _int_game_id(game.game_id or "")
    want_bc = "bc" in bundle_tasks
    want_calls = "calls" in bundle_tasks
    want_schupfen = "schupfen" in bundle_tasks
    want_belief = "belief" in bundle_tasks
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
            team_totals=team_totals,
        ))
        rid = parsed_round.round_index
        if want_bc:
            bc_examples.extend(_emit_bc_examples_for_round(
                parsed_round, replay,
                skill_lookup=skill_lookup,
                sample_weight=sample_weight,
                team_totals=team_totals,
                neutral_decile=neutral_decile,
            ))
        if want_calls:
            # call_emit sets game_id (str, the split key) + round_id + guards
            # a None skill itself, so no _stamp_provenance pass is needed here.
            _gid = game.game_id or ""
            call_tichu_examples.extend(tichu_examples_for_round(
                parsed_round, replay,
                skill_lookup=skill_lookup,
                neutral_decile=neutral_decile,
                sample_weight=sample_weight,
                game_id=_gid, round_id=rid,
            ))
            call_grand_examples.extend(grand_tichu_examples_for_round(
                parsed_round,
                skill_lookup=skill_lookup,
                neutral_decile=neutral_decile,
                sample_weight=sample_weight,
                game_id=_gid, round_id=rid,
            ))
        if want_schupfen:
            schupfen_examples.extend(_stamp_provenance(
                _schupfen_examples_for_round(
                    parsed_round,
                    skill_lookup=skill_lookup,
                    neutral_decile=neutral_decile,
                    sample_weight=sample_weight,
                ),
                game_id_int, rid, neutral_decile,
            ))
        if want_belief:
            # BeliefExample is frozen; provenance is set at construction
            # (no post-hoc stamping). Labels are read from the same replay.
            belief_examples.extend(belief_examples_for_round(
                parsed_round, replay, game_id=game_id_int, round_id=rid,
            ))
    return _GameResult(
        game_id=game_id,
        records=records,
        rounds_total=rounds_total,
        rounds_matched=rounds_matched,
        failures=failures,
        bc_examples=bc_examples,
        call_tichu_examples=call_tichu_examples,
        call_grand_examples=call_grand_examples,
        schupfen_examples=schupfen_examples,
        belief_examples=belief_examples,
        is_complete_game=team_totals is not None,
    )


def _parse_and_process(
    pair: tuple[str, str],
    *,
    skill_lookup: dict[str, int | None],
    recency_cutoff_game_id: int,
    recency_weight: float,
    bundle_tasks: frozenset[str] = frozenset(),
) -> _GameResult:
    """Worker for `stream_raw_to_parquet`: parse `.tch` text + replay + emit
    in one process boundary crossing. Moves the parse cost off the dispatcher
    and halves the per-game pickle size (raw text < pickled ParsedGame)."""
    game_id, text = pair
    try:
        game = parse_tch(text, game_id=game_id)
    except Exception as exc:  # noqa: BLE001 — surface every parse failure
        return _GameResult(
            game_id=game_id,
            records=[],
            rounds_total=0,
            rounds_matched=0,
            failures=[],
            parse_error=str(exc),
        )
    return _process_game(
        game,
        skill_lookup=skill_lookup,
        recency_cutoff_game_id=recency_cutoff_game_id,
        recency_weight=recency_weight,
        bundle_tasks=bundle_tasks,
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
    # Complete Game vs Incomplete Session split (parsed games only — parse
    # failures contribute to neither). Drives the A/B comparison's report on
    # what fraction of corpus the AWR game-outcome target actually fits on.
    complete_games: int = 0
    incomplete_sessions: int = 0
    rounds_total: int = 0
    rounds_matched: int = 0
    row_counts: dict[str, int] = field(default_factory=lambda: {t: 0 for t in _KNOWN_DECISION_TYPES})
    failed_game_ids: list[str] = field(default_factory=list)
    round_failures: list[RoundFailure] = field(default_factory=list)
    # Populated only by `stream_raw_to_parquet` (parse runs in workers).
    # `stream_to_parquet` callers parse upstream and surface failures themselves.
    parse_failures: list[str] = field(default_factory=list)


def stream_to_parquet(
    games: Iterable[ParsedGame],
    output_dir: Path,
    *,
    ratings_path: str | Path | None = None,
    recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
    recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
    rows_per_flush: int = _DEFAULT_ROWS_PER_FLUSH,
    workers: int = 1,
    bundle_dirs: Mapping[str, Path] | None = None,
    on_game_done: Callable[["StreamStats"], None] | None = None,
) -> StreamStats:
    """Stream pre-parsed games to per-decision Parquet shards, filtering at
    the Round granularity (see ADR-0009).

    With `bundle_dirs` set (a `{task: out_dir}` map), the same replay pass also
    writes one materialised bundle per task — consolidating what used to be a
    separate replay per trainer into one pass (ADR-0020). See the 2026-05-29
    pipeline-perf handoff and ADR-0016.

    Prefer `stream_raw_to_parquet` for production runs over `.tch` archives:
    it parses inside workers, which removes parse_tch from the serial
    dispatcher path (the dominant bottleneck at workers >= 4) and halves the
    per-game pickle size.
    """
    skill_lookup = _load_skill_lookup(ratings_path) if ratings_path else {}
    worker = partial(
        _process_game,
        skill_lookup=skill_lookup,
        recency_cutoff_game_id=recency_cutoff_game_id,
        recency_weight=recency_weight,
        bundle_tasks=frozenset(bundle_dirs or ()),
    )
    return _run_stream(
        games, output_dir, worker,
        rows_per_flush=rows_per_flush,
        workers=workers,
        bundle_dirs=bundle_dirs,
        on_game_done=on_game_done,
    )


def stream_raw_to_parquet(
    pairs: Iterable[tuple[str, str]],
    output_dir: Path,
    *,
    ratings_path: str | Path | None = None,
    recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
    recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
    rows_per_flush: int = _DEFAULT_ROWS_PER_FLUSH,
    workers: int = 1,
    bundle_dirs: Mapping[str, Path] | None = None,
    on_game_done: Callable[["StreamStats"], None] | None = None,
) -> StreamStats:
    """Stream raw `(game_id, tch_text)` pairs to per-decision Parquet shards,
    parsing inside the worker process. Parse failures are recorded in
    `StreamStats.parse_failures` and surface via `on_game_done` like any
    other completed game (so the progress bar advances).

    With `bundle_dirs` set (a `{task: out_dir}` map), the same replay pass also
    writes one materialised bundle per task (ADR-0020)."""
    skill_lookup = _load_skill_lookup(ratings_path) if ratings_path else {}
    worker = partial(
        _parse_and_process,
        skill_lookup=skill_lookup,
        recency_cutoff_game_id=recency_cutoff_game_id,
        recency_weight=recency_weight,
        bundle_tasks=frozenset(bundle_dirs or ()),
    )
    return _run_stream(
        pairs, output_dir, worker,
        rows_per_flush=rows_per_flush,
        workers=workers,
        bundle_dirs=bundle_dirs,
        on_game_done=on_game_done,
    )


def _run_stream(
    source: Iterable,
    output_dir: Path,
    worker: Callable,
    *,
    rows_per_flush: int,
    workers: int,
    bundle_dirs: Mapping[str, Path] | None = None,
    on_game_done: Callable[["StreamStats"], None] | None,
) -> StreamStats:
    """Shared dispatch loop. `worker` consumes one source item and returns a
    `_GameResult` (possibly with `parse_error` set).

    One `ParquetWriter` is opened lazily per decision type the first time a
    row of that type is buffered, and closed in `try/finally` so partial runs
    leave valid (footer-written) Parquet files. After the iterator drains,
    any decision types that never saw a record get an empty file written so
    the on-disk shape is invariant.

    When `bundle_out_dir` is set, each result's `bc_examples` are streamed
    into `materialise()` as the results arrive — the writer pulls examples
    incrementally rather than the dispatcher buffering the whole corpus, so
    peak RAM stays bounded by `materialise`'s chunk size regardless of how
    many games are processed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        if result.parse_error is not None:
            stats.parse_failures.append(result.game_id)
        else:
            stats.rounds_total += result.rounds_total
            stats.rounds_matched += result.rounds_matched
            if result.had_failure:
                stats.games_with_failed_rounds += 1
                stats.failed_game_ids.append(result.game_id)
            else:
                stats.games_fully_matched += 1
            if result.is_complete_game:
                stats.complete_games += 1
            else:
                stats.incomplete_sessions += 1
            stats.round_failures.extend(result.failures)
            for record in result.records:
                buffers[record.decision_type].append(record)
                stats.row_counts[record.decision_type] += 1
            for decision_type, buf in buffers.items():
                if len(buf) >= rows_per_flush:
                    _flush(decision_type)
        if on_game_done is not None:
            on_game_done(stats)

    def _iter_results():
        """Yield one `_GameResult` per source item as it completes."""
        if workers <= 1:
            for item in source:
                yield worker(item)
        else:
            # Sliding-window submit + wait so `on_game_done` fires per game (not
            # per `pool.map` chunk) and the input iterator isn't drained upfront.
            source_iter = iter(source)
            pending: set = set()
            max_pending = workers * 2

            def _top_up(n: int) -> None:
                for _ in range(n):
                    try:
                        item = next(source_iter)
                    except StopIteration:
                        return
                    pending.add(pool.submit(worker, item))

            with ProcessPoolExecutor(max_workers=workers) as pool:
                _top_up(max_pending)
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        yield fut.result()
                    _top_up(len(done))

    # One push-style bundle writer per requested task (ADR-0020). Each
    # consumes the per-task examples carried on every `_GameResult` as the
    # results arrive — peak RAM stays bounded by each writer's chunk size,
    # not the corpus size. Deferred imports: the writers pull in numpy + the
    # bc layout, unwanted on a manifest-only run.
    bundle_dirs = dict(bundle_dirs or {})
    bundle_writers: dict[str, object] = {}
    if bundle_dirs:
        from tichu_training.bc.call_materialised import CallBundleWriter
        from tichu_training.bc.materialised import BCBundleWriter
        from tichu_training.bc.schupfen_materialised import SchupfenBundleWriter
        from tichu_training.belief.belief_materialised import BeliefBundleWriter
        if "bc" in bundle_dirs:
            bundle_writers["bc"] = BCBundleWriter(bundle_dirs["bc"])
        if "calls" in bundle_dirs:
            bundle_writers["calls"] = CallBundleWriter(bundle_dirs["calls"])
        if "schupfen" in bundle_dirs:
            bundle_writers["schupfen"] = SchupfenBundleWriter(bundle_dirs["schupfen"])
        if "belief" in bundle_dirs:
            bundle_writers["belief"] = BeliefBundleWriter(bundle_dirs["belief"])

    try:
        for result in _iter_results():
            _apply(result)
            bc_w = bundle_writers.get("bc")
            if bc_w is not None:
                bc_w.add_many(result.bc_examples)
            calls_w = bundle_writers.get("calls")
            if calls_w is not None:
                calls_w.add_many("call_tichu", result.call_tichu_examples)
                calls_w.add_many("call_grand_tichu", result.call_grand_examples)
            schupfen_w = bundle_writers.get("schupfen")
            if schupfen_w is not None:
                schupfen_w.add_many(result.schupfen_examples)
            belief_w = bundle_writers.get("belief")
            if belief_w is not None:
                belief_w.add_many(result.belief_examples)
        for decision_type in _KNOWN_DECISION_TYPES:
            _flush(decision_type)
            if decision_type not in writers:
                path = output_dir / f"{decision_type}_00000.parquet"
                writers[decision_type] = pq.ParquetWriter(path, _SCHEMA)
    finally:
        for w in writers.values():
            w.close()
        for bw in bundle_writers.values():
            bw.close()

    return stats


def _emit_records_for_round(
    game: ParsedGame,
    parsed_round: ParsedRound,
    replay: ReplayResult,
    *,
    skill_lookup: dict[str, int | None],
    sample_weight: float,
    team_totals: tuple[int, int] | None,
):
    team_outcome = parsed_round.ergebnis[0] - parsed_round.ergebnis[1]
    for parsed_action, _engine_action in replay.decisions:
        decision_type = _DECISION_TYPE_BY_KIND.get(parsed_action.kind)
        if decision_type is None:
            continue
        player = parsed_action.player
        if 0 <= player < 4:
            # Per-round handles capture mid-game substitutions and anonymous
            # seats correctly; the previous game-level snapshot mis-attributed
            # post-substitution decisions to the round-0 handle. See ADR-0010.
            handle = parsed_round.handles[player]
        else:
            handle = ""
        team = player % 2 if 0 <= player < 4 else 0
        round_won = (
            parsed_round.ergebnis[team] > parsed_round.ergebnis[1 - team]
        )
        if team_totals is None:
            game_won = None
        else:
            game_won = team_totals[team] > team_totals[1 - team]
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
            game_won=game_won,
            featurizer_version=FEATURIZER_VERSION,
            action_space_version=ACTION_SPACE_VERSION,
            skill_decile=skill_lookup.get(handle),
            sample_weight=sample_weight,
        )


def _emit_bc_examples_for_round(
    parsed_round: ParsedRound,
    replay: ReplayResult,
    *,
    skill_lookup: dict[str, int | None],
    sample_weight: float,
    team_totals: tuple[int, int] | None,
    neutral_decile: int,
):
    """Yield one `BCExample` per BC-relevant decision in a validated round.

    Mirrors `ParquetBCDataset.__iter__` (and `_worker_loop`) exactly: same
    decision-type filter, same featurise/target/legal-mask logic, same skip
    conditions, same field values. Keeping this in lockstep is what makes the
    consolidated bundle byte-identical to the parquet -> materialise bundle.
    """
    from tichu_training.action_space import bc_target_for_concrete, legal_mask
    from tichu_training.featurizer import featurize
    from tichu_engine.state import DragonGivePending

    team_outcome = float(parsed_round.ergebnis[0] - parsed_round.ergebnis[1])
    for (parsed_action, concrete), pre_state, cached_actions in zip(
        replay.decisions,
        replay.pre_decision_states,
        replay.legal_actions_at,
    ):
        if pre_state is None:
            continue  # Tichu/Grand-Tichu passthrough or phantom pass
        decision_type = _KIND_TO_DECISION_TYPE.get(parsed_action.kind)
        if decision_type is None:
            continue  # schupfen, tichu, grand_tichu — not BC's job
        player = parsed_action.player
        if not 0 <= player < 4:
            continue
        private = pre_state.private_view(player)
        features = featurize(private)
        if decision_type == "dragon_assignment":
            pending = pre_state.public.pending_decision
            if not isinstance(pending, DragonGivePending):
                continue
            try:
                target = bc_target_for_concrete(
                    decision_type, concrete, winner_seat=pending.winner,
                )
            except ValueError:
                continue
        else:
            try:
                target = bc_target_for_concrete(decision_type, concrete)
            except ValueError:
                continue
        mask = legal_mask(
            decision_type, pre_state, player, cached_actions=cached_actions,
        )
        if not mask[target]:
            continue
        handle = parsed_round.handles[player]
        skill = skill_lookup.get(handle, neutral_decile)
        if skill is None:
            skill = neutral_decile
        team = player % 2
        if team_totals is None:
            game_won: bool | None = None
        else:
            game_won = team_totals[team] > team_totals[1 - team]
        yield BCExample(
            decision_type=decision_type,
            features=features,
            target=int(target),
            legal_mask=mask,
            sample_weight=sample_weight,
            skill_decile=skill,
            round_outcome=team_outcome,
            game_won=game_won,
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
        columns["game_won"].append(r.game_won)
        columns["featurizer_version"].append(r.featurizer_version)
        columns["action_space_version"].append(r.action_space_version)
        columns["skill_decile"].append(r.skill_decile)
        columns["sample_weight"].append(r.sample_weight)
    return pa.Table.from_pydict(columns, schema=_SCHEMA)
