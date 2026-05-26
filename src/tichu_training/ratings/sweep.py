"""Sweep parsed BSW games chronologically and update TrueSkill ratings.

Teams are fixed by seat: seats 0+2 = team A, seats 1+3 = team B. A game's
outcome is the sign of the summed Ergebnis across its rounds; ties skip the
update entirely (player still gets registered with default rating and
n_games == 0).
"""

from dataclasses import dataclass
from typing import Iterable

import trueskill

from tichu_training.bsw.records import ParsedGame


@dataclass
class PlayerRating:
    handle: str
    rating: trueskill.Rating
    n_games: int

    @property
    def mu(self) -> float:
        return self.rating.mu

    @property
    def sigma(self) -> float:
        return self.rating.sigma


def update_for_game(
    game: ParsedGame,
    ratings: dict[str, PlayerRating],
    env: trueskill.TrueSkill,
) -> None:
    """Apply one game's TrueSkill update into `ratings` in place.

    Used by the streaming CLI so the full `parsed_games` list never has to
    be held in memory. `compute_ratings` is the corresponding loop wrapper.
    """
    if not game.rounds:
        return
    # ADR-0010: TrueSkill demands per-identified-stable-game seating.
    # Skip games where any round contains an Anonymous Seat (empty
    # handle) OR where any seat's handle changes between rounds (BSW
    # mid-game player substitution). Those games still feed BC training
    # — substituted seats get the round's handle; anonymous seats get
    # Neutral Skill Decile — but never update player ratings.
    round_handles = [r.handles for r in game.rounds]
    if any("" in r_h for r_h in round_handles):
        return
    if any(r_h != round_handles[0] for r_h in round_handles[1:]):
        return
    # Seating is stable across rounds (guarded above), so the round-0
    # handles are the canonical identities for this game.
    handles = round_handles[0]
    team_a_score = sum(r.ergebnis[0] for r in game.rounds)
    team_b_score = sum(r.ergebnis[1] for r in game.rounds)
    # Make sure every participant is registered even on a draw.
    players = []
    for h in handles:
        if h not in ratings:
            ratings[h] = PlayerRating(handle=h, rating=env.create_rating(), n_games=0)
        players.append(ratings[h])
    if team_a_score == team_b_score:
        return
    team_a = [players[0].rating, players[2].rating]
    team_b = [players[1].rating, players[3].rating]
    # ranks=[0,1] means team A first (winner); [1,0] means team B first.
    ranks = [0, 1] if team_a_score > team_b_score else [1, 0]
    new_a, new_b = env.rate([team_a, team_b], ranks=ranks)
    players[0].rating, players[2].rating = new_a
    players[1].rating, players[3].rating = new_b
    for p in players:
        p.n_games += 1


def compute_ratings(
    games: Iterable[ParsedGame],
    env: trueskill.TrueSkill | None = None,
) -> dict[str, PlayerRating]:
    """Apply per-game TrueSkill updates and return one rating per handle."""
    env = env or trueskill.TrueSkill()
    ratings: dict[str, PlayerRating] = {}
    for game in games:
        update_for_game(game, ratings, env)
    return ratings
