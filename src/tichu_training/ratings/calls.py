"""Per-player Tichu / Grand Tichu call success counters.

A Tichu (or Grand Tichu) call is credited as successful for the caller if
their team's net round score (`ergebnis[team] - ergebnis[1-team]`) is at
least +100. This matches the points swing produced by a successful call
and is the proxy permitted by the spec when engine replay output is not
re-derived here.
"""

from dataclasses import dataclass
from typing import Iterable

from tichu_training.bsw.records import ParsedGame


_SUCCESS_MARGIN = 100


@dataclass
class CallStats:
    handle: str
    tichu_calls: int = 0
    tichu_wins: int = 0
    grand_tichu_calls: int = 0
    grand_tichu_wins: int = 0

    @property
    def tichu_success_rate(self) -> float | None:
        if self.tichu_calls == 0:
            return None
        return self.tichu_wins / self.tichu_calls

    @property
    def grand_tichu_success_rate(self) -> float | None:
        if self.grand_tichu_calls == 0:
            return None
        return self.grand_tichu_wins / self.grand_tichu_calls


def compute_call_stats(games: Iterable[ParsedGame]) -> dict[str, CallStats]:
    """Sweep games and emit per-handle Tichu/Grand Tichu success counts.

    Players who never called either kind are omitted from the result.
    """
    stats: dict[str, CallStats] = {}

    def _get(handle: str) -> CallStats:
        if handle not in stats:
            stats[handle] = CallStats(handle=handle)
        return stats[handle]

    for game in games:
        handles = game.handles
        if len(handles) != 4:
            continue
        for r in game.rounds:
            margin_per_team = (
                r.ergebnis[0] - r.ergebnis[1],
                r.ergebnis[1] - r.ergebnis[0],
            )
            for seat in r.tichu_callers:
                if not 0 <= seat < 4:
                    continue
                cs = _get(handles[seat])
                cs.tichu_calls += 1
                if margin_per_team[seat % 2] >= _SUCCESS_MARGIN:
                    cs.tichu_wins += 1
            for seat in r.grand_tichu_callers:
                if not 0 <= seat < 4:
                    continue
                cs = _get(handles[seat])
                cs.grand_tichu_calls += 1
                if margin_per_team[seat % 2] >= _SUCCESS_MARGIN:
                    cs.grand_tichu_wins += 1
    return stats
