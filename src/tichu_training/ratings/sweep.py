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


def compute_ratings(
    games: Iterable[ParsedGame],
    env: trueskill.TrueSkill | None = None,
) -> dict[str, PlayerRating]:
    """Apply per-game TrueSkill updates and return one rating per handle."""
    env = env or trueskill.TrueSkill()
    ratings: dict[str, PlayerRating] = {}

    def _get(handle: str) -> PlayerRating:
        if handle not in ratings:
            ratings[handle] = PlayerRating(handle=handle, rating=env.create_rating(), n_games=0)
        return ratings[handle]

    for game in games:
        handles = game.handles
        if len(handles) != 4:
            continue
        team_a_score = sum(r.ergebnis[0] for r in game.rounds)
        team_b_score = sum(r.ergebnis[1] for r in game.rounds)
        # Make sure every participant is registered even on a draw.
        players = [_get(h) for h in handles]
        if team_a_score == team_b_score:
            continue
        team_a = [players[0].rating, players[2].rating]
        team_b = [players[1].rating, players[3].rating]
        # ranks=[0,1] means team A first (winner); [1,0] means team B first.
        ranks = [0, 1] if team_a_score > team_b_score else [1, 0]
        new_a, new_b = env.rate([team_a, team_b], ranks=ranks)
        players[0].rating, players[2].rating = new_a
        players[1].rating, players[3].rating = new_b
        for p in players:
            p.n_games += 1
    return ratings
