"""Behavioral Drift Benchmark — Synthetic Games.

A **Synthetic Game** chains benchmark Round results, in order, until a team
reaches 1000 (the higher total wins; a tie plays on). It is exact only for
**score-blind** Agents — every current net sees round-local scores only
(`round_local_team_scores`), so a Round plays out identically at 0–0 and 950–900
and a Game is a sequence of independent Rounds. A score-aware Checkpoint breaks
that and needs a real game loop. See CONTEXT.md §"Synthetic Game".

**One Round per deal.** A deal's two Seat-Swap halves are the same cards with
the teams swapped — near-mirror images (exact mirrors in the BC arm, measured
corr −1.0; −0.48 in the Subject arm). Chaining both made consecutive Rounds
cancel and Games run ~1.2 Rounds long (BC 10.67 vs 9.49 in human BSW games); a
real Game never replays a deal. So each deal contributes the half `deal % 2`,
alternating which team the Subject sits in.

Game metrics are means over Games (reported in `rate`, with the Game count in
`opportunities`), but their bootstrap resamples deals *as a sequence* and
re-chains the Games on every replicate, so they have their own summariser. The
same deal draw serves both arms, which pairs the Δ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tichu_eval.drift_stats import ARMS, _ci, _two_sided_p

TARGET = 1000
CELLS = ("rounds_per_game", "win_rate", "margin")


@dataclass(frozen=True)
class Game:
    rounds: int
    subject_won: bool
    margin: int       # Subject total − opponent total at the end


def synthetic_games(round_totals) -> list[Game]:
    """Chain `(subject_total, opponent_total)` Round results into Games. A
    trailing Game that never reaches 1000 is dropped."""
    games: list[Game] = []
    s = o = n = 0
    for subject_total, opponent_total in round_totals:
        s += subject_total
        o += opponent_total
        n += 1
        if max(s, o) >= TARGET and s != o:
            games.append(Game(rounds=n, subject_won=s > o, margin=s - o))
            s = o = n = 0
    return games


def summarise_games(rounds: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Game metrics per arm + paired Δ, in `summarise_cells`' row schema."""
    deals = np.array(sorted(set(rounds.deal)))
    per_arm = {arm: _deal_sequences(rounds[rounds.arm == arm], deals) for arm in ARMS}
    point_games = {arm: synthetic_games(_chain(per_arm[arm], range(len(deals)))) for arm in ARMS}
    point = {arm: _stats(games) for arm, games in point_games.items()}
    rng = np.random.default_rng(seed)
    boot = {arm: np.empty((n_boot, len(CELLS))) for arm in ARMS}
    for b in range(n_boot):
        draw = rng.integers(0, len(deals), size=len(deals))   # shared by both arms
        for arm in ARMS:
            boot[arm][b] = _stats(synthetic_games(_chain(per_arm[arm], draw)))

    out = []
    for c, cell in enumerate(CELLS):
        for arm in ARMS:
            out.append(_row(cell, arm, point[arm][c], boot[arm][:, c],
                            n_games=len(point_games[arm])))
        d_bs = boot["subject"][:, c] - boot["bc"][:, c]
        out.append(_row(cell, "delta", point["subject"][c] - point["bc"][c], d_bs,
                        p=_two_sided_p(d_bs)))
    return pd.DataFrame(out)


def _deal_sequences(arm_rounds: pd.DataFrame, deals) -> list[list[tuple[int, int]]]:
    """Per deal, its one chained Round — Seat-Swap half `deal % 2` — as
    (Subject team total, opponent total)."""
    by_deal: dict = {d: [] for d in deals}
    one_half = arm_rounds[arm_rounds.half == arm_rounds.deal % 2]
    for r in one_half.sort_values(["deal", "half"]).itertuples():
        mine = r.total_0 if r.subject_team == 0 else r.total_1
        theirs = r.total_1 if r.subject_team == 0 else r.total_0
        by_deal[r.deal].append((int(mine), int(theirs)))
    return [by_deal[d] for d in deals]


def _chain(sequences, order):
    for i in order:
        yield from sequences[i]


def _stats(games: list[Game]) -> np.ndarray:
    if not games:
        return np.full(len(CELLS), np.nan)
    return np.array([
        np.mean([g.rounds for g in games]),
        np.mean([g.subject_won for g in games]),
        np.mean([g.margin for g in games]),
    ])


def _row(cell, arm, value, samples, p=np.nan, n_games=np.nan) -> dict:
    lo, hi = _ci(samples)
    return {"cell": cell, "arm": arm, "events": np.nan, "opportunities": n_games,
            "exposure": np.nan, "exposure_lo": np.nan, "exposure_hi": np.nan,
            "rate": value, "rate_lo": lo, "rate_hi": hi, "p": p, "suppressed": False}


def reference_games(rounds: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Rounds per Game for an unpaired reference (the human games): the REAL
    length of each Game that reached 1000 (`game_length`), bootstrapped by Game.
    Win rate and margin are not reported — both teams are the same population."""
    lengths = rounds.drop_duplicates("game").game_length.dropna().to_numpy(dtype=np.float64)
    arm = rounds.arm.iloc[0]
    if not len(lengths):
        return pd.DataFrame([_row("rounds_per_game", arm, np.nan, np.full(n_boot, np.nan), n_games=0)])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(lengths), size=(n_boot, len(lengths)))
    row = _row("rounds_per_game", arm, float(lengths.mean()), lengths[draws].mean(axis=1),
               n_games=len(lengths))
    return pd.DataFrame([{**row, "bh_survives": False}])
