"""Emit per-decision training records as Parquet, sharded by `decision_type`.

Each row corresponds to one decision a BSW player made during a game. The
columns follow the v1 PRD schema:

  decision_type, game_id, round_id, timestamp, player_handle, action_taken,
  state, legal_actions_mask, round_outcome, round_won,
  featurizer_version, action_space_version, skill_decile, sample_weight

`state` and `legal_actions_mask` are reserved (filled at training-loop load
time, not at parse time). `featurizer_version` / `action_space_version` are
stamped from the live module constants. `skill_decile` is joined from a
ratings table if one is provided. `sample_weight` is `1.0` for games whose
`game_id >= recency_cutoff_game_id` (default `1855844`, first game of 2015),
else `recency_weight` (default `0.5`).

Output layout: one shard per decision type, deterministically named
`{decision_type}_00000.parquet` so training jobs can glob shards without a
manifest. (Multi-shard splits can fill in higher suffixes later.)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_engine.cards import Card, SpecialCard
from tichu_engine.combinations import CardOrSpecial
from tichu_training.action_space import ACTION_SPACE_VERSION
from tichu_training.bsw.records import ParsedAction, ParsedGame
from tichu_training.bsw.replay import replay_round
from tichu_training.featurizer import FEATURIZER_VERSION


# Map ParsedAction.kind -> decision_type emitted to Parquet.
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

# First BSW game id of 2015. Games at or after this id are post-2015.
_DEFAULT_RECENCY_CUTOFF: int = 1855844
_DEFAULT_RECENCY_WEIGHT: float = 0.5


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
    round_outcome: int      # team-0 minus team-1 Ergebnis for the round
    round_won: bool         # did the acting player's team have the higher score this round?
    featurizer_version: str | None
    action_space_version: str | None
    skill_decile: int | None
    sample_weight: float


def write_parquet_shards(
    games: Iterable[ParsedGame],
    output_dir: Path,
    *,
    ratings_path: str | Path | None = None,
    recency_cutoff_game_id: int = _DEFAULT_RECENCY_CUTOFF,
    recency_weight: float = _DEFAULT_RECENCY_WEIGHT,
) -> dict[str, int]:
    """Write per-decision-type Parquet shards. Returns row counts per type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    skill_lookup = _load_skill_lookup(ratings_path) if ratings_path else {}
    buckets: dict[str, list[_Record]] = {t: [] for t in _KNOWN_DECISION_TYPES}
    for record in _emit_records(
        games,
        skill_lookup=skill_lookup,
        recency_cutoff_game_id=recency_cutoff_game_id,
        recency_weight=recency_weight,
    ):
        buckets[record.decision_type].append(record)
    counts: dict[str, int] = {}
    for decision_type, records in buckets.items():
        path = output_dir / f"{decision_type}_00000.parquet"
        table = _to_table(records)
        pq.write_table(table, path)
        counts[decision_type] = len(records)
    return counts


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


def _emit_records(
    games: Iterable[ParsedGame],
    *,
    skill_lookup: dict[str, int | None],
    recency_cutoff_game_id: int,
    recency_weight: float,
):
    for game in games:
        weight = _sample_weight_for(game.game_id or "", recency_cutoff_game_id, recency_weight)
        for parsed_round in game.rounds:
            replay = replay_round(parsed_round)
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
                    sample_weight=weight,
                )


def _serialise_action(action: ParsedAction) -> str:
    """A compact string form of the action so it round-trips through Parquet.

    A full structured action representation comes with the action-space module
    (#006); for now the parsed kind plus the relevant payload is enough to
    uniquely identify which decision was made.
    """
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
    schema = pa.schema([
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
    columns: dict[str, list] = {f.name: [] for f in schema}
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
    return pa.Table.from_pydict(columns, schema=schema)
