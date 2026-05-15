"""Emit per-decision training records as Parquet, sharded by `decision_type`.

Each row corresponds to one decision a BSW player made during a game. The
fields are the ones listed in the project PRD; `state` and `legal_actions_mask`
are reserved for the featurizer (issue #006) and currently emitted as `null`.

Output layout: one Parquet file per decision type under the given directory,
e.g. `output/play.parquet`, `output/pass_card.parquet`, ...
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from tichu_engine.cards import Card, SpecialCard
from tichu_engine.combinations import CardOrSpecial
from tichu_training.bsw.records import ParsedAction, ParsedGame
from tichu_training.bsw.replay import replay_round


# Map ParsedAction.kind -> decision_type emitted to Parquet.
_DECISION_TYPE_BY_KIND: dict[str, str] = {
    "play": "play",
    "pass": "play",
    "schupfen": "pass_card",
    "tichu": "call_tichu",
    "grand_tichu": "call_grand_tichu",
    "wish": "wish_rank",
    "dragon_give": "dragon_give",
}


_KNOWN_DECISION_TYPES = ("play", "pass_card", "call_tichu", "call_grand_tichu", "wish_rank", "dragon_give")


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


def write_parquet_shards(
    games: Iterable[ParsedGame], output_dir: Path
) -> dict[str, int]:
    """Write per-decision-type Parquet shards. Returns row counts per type."""
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[_Record]] = {t: [] for t in _KNOWN_DECISION_TYPES}
    for record in _emit_records(games):
        buckets[record.decision_type].append(record)
    counts: dict[str, int] = {}
    for decision_type, records in buckets.items():
        path = output_dir / f"{decision_type}.parquet"
        table = _to_table(records)
        pq.write_table(table, path)
        counts[decision_type] = len(records)
    return counts


def _emit_records(games: Iterable[ParsedGame]):
    for game in games:
        for parsed_round in game.rounds:
            replay = replay_round(parsed_round)
            team_outcome = parsed_round.ergebnis[0] - parsed_round.ergebnis[1]
            for parsed_action, _engine_action in replay.decisions:
                decision_type = _DECISION_TYPE_BY_KIND.get(parsed_action.kind)
                if decision_type is None:
                    continue
                player = parsed_action.player
                # `player` is the seat (0..3). Handle is in game.handles[player].
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
                    state=None,                # filled by featurizer (issue #006)
                    legal_actions_mask=None,   # filled by action-space module (#006)
                    round_outcome=team_outcome,
                    round_won=round_won,
                    featurizer_version=None,
                    action_space_version=None,
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
    return pa.Table.from_pydict(columns, schema=schema)
