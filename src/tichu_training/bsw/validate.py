"""Corpus-scale replay validation.

`validate_game(parsed_game)` replays every round of a parsed game and reports
which rounds' engine scores match the recorded BSW `Ergebnis`. The result is a
`GameValidationResult` summarising matches, mismatches, and any replay errors.

`validate_corpus(parsed_games)` aggregates across many games and produces a
pass-rate statistic plus a list of failing game IDs suitable for writing to
`known_bad_games.txt`.
"""

from dataclasses import dataclass

from tichu_training.bsw.records import ParsedGame
from tichu_training.bsw.replay import ReplayResult, replay_round


@dataclass(frozen=True)
class RoundValidation:
    round_index: int
    expected: tuple[int, int]
    actual: tuple[int, int] | None
    matches: bool
    illegal_action: bool


@dataclass(frozen=True)
class GameValidationResult:
    game_id: str | None
    rounds: tuple[RoundValidation, ...]

    @property
    def fully_matches(self) -> bool:
        return all(r.matches for r in self.rounds)

    @property
    def any_illegal(self) -> bool:
        return any(r.illegal_action for r in self.rounds)


@dataclass(frozen=True)
class CorpusValidationStats:
    games_total: int
    games_matched: int
    games_failed: int
    games_with_illegal_actions: int
    failed_game_ids: tuple[str, ...]

    @property
    def pass_rate(self) -> float:
        if self.games_total == 0:
            return 0.0
        return self.games_matched / self.games_total


def validate_game(parsed: ParsedGame) -> GameValidationResult:
    rounds: list[RoundValidation] = []
    for r in parsed.rounds:
        replay = replay_round(r)
        if replay.final_state is None:
            rounds.append(RoundValidation(
                round_index=r.round_index,
                expected=r.ergebnis,
                actual=None,
                matches=False,
                illegal_action=True,
            ))
            continue
        actual = replay.final_state.public.scores
        rounds.append(RoundValidation(
            round_index=r.round_index,
            expected=r.ergebnis,
            actual=actual,
            matches=actual == r.ergebnis,
            illegal_action=False,
        ))
    return GameValidationResult(game_id=parsed.game_id, rounds=tuple(rounds))


def validate_corpus(parsed_games: list[ParsedGame]) -> CorpusValidationStats:
    matched = 0
    failed: list[str] = []
    illegal = 0
    for g in parsed_games:
        result = validate_game(g)
        if result.any_illegal:
            illegal += 1
        if result.fully_matches:
            matched += 1
        else:
            failed.append(g.game_id or "<unknown>")
    return CorpusValidationStats(
        games_total=len(parsed_games),
        games_matched=matched,
        games_failed=len(failed),
        games_with_illegal_actions=illegal,
        failed_game_ids=tuple(failed),
    )
