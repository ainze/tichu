"""Behavioral Drift Benchmark — the metric catalogue.

A `Metric` is a pure function `DriftLog -> counts` where `counts` has columns
`deal, arm, cell, events, opportunities`: per-deal tallies for each arm and each
cell (one cell for a plain rate, one per category for a split or distribution).
Metrics read **Subject seats only** (`is_subject`) — the opponents are the same
BC team in both arms — and Play metrics skip **Forced Play Decisions**, which
carry no choice. `summarise` runs the whole panel through `drift_stats` and
flags which Δs survive Benjamini–Hochberg across it. Adding a metric is adding a
function here; the log never needs replaying. See CONTEXT.md §"Behavioral Drift
Benchmark".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tichu_eval.drift_games import reference_games, summarise_games
from tichu_eval.drift_stats import benjamini_hochberg, summarise_cells, summarise_reference

_COUNT_COLUMNS = ["deal", "arm", "cell", "events", "opportunities"]


@dataclass(frozen=True)
class Metric:
    name: str
    family: str
    description: str
    fn: Callable[..., pd.DataFrame]
    # Unconditional metrics (per-Round outcomes) have no situation to be exposed
    # to — their "opportunity" is the Round itself — so the report omits Exposure.
    conditional: bool = True


# --- Tallying helpers ---------------------------------------------------------

def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=_COUNT_COLUMNS)


def _column(frame: pd.DataFrame, spec):
    """`spec` is a column name or a Series aligned to `frame`."""
    return (frame[spec] if isinstance(spec, str) else spec).to_numpy()


def _tally(frame: pd.DataFrame, cell, event) -> pd.DataFrame:
    """Per-deal counts of a boolean `event` over `frame`'s rows, by `cell`."""
    return _tally_sum(frame, cell, pd.Series(_column(frame, event), index=frame.index).astype(int))


def _tally_sum(frame: pd.DataFrame, cell, value) -> pd.DataFrame:
    """Per-deal sums of a numeric `value` over `frame`'s rows (opportunities =
    rows) — a boolean event, or a quantity such as points whose ratio of sums is
    a per-Round mean."""
    if frame.empty:
        return _empty()
    t = pd.DataFrame({
        "deal": frame.deal.to_numpy(), "arm": frame.arm.to_numpy(),
        "cell": _column(frame, cell), "value": _column(frame, value).astype(float),
    })
    g = t.groupby(["deal", "arm", "cell"], sort=False).value.agg(["sum", "size"])
    return g.rename(columns={"sum": "events", "size": "opportunities"}).reset_index()


def _distribution(frame: pd.DataFrame, category, categories=None) -> pd.DataFrame:
    """Per-deal counts for a categorical metric: every category is an event over
    **all** of that arm's decisions in the deal, so each (deal, arm) carries every
    category — a zero-event row still contributes its denominator. `categories`
    fixes the category list (so a never-seen category still reports 0%);
    otherwise it is every category seen in either arm."""
    if frame.empty:
        return _empty()
    t = pd.DataFrame({"deal": frame.deal.to_numpy(), "arm": frame.arm.to_numpy(),
                      "cell": _column(frame, category)})
    events = t.groupby(["deal", "arm", "cell"]).size().rename("events")
    totals = t.groupby(["deal", "arm"]).size().rename("opportunities").reset_index()
    cells = pd.DataFrame({"cell": list(categories) if categories is not None
                          else sorted(set(t.cell), key=str)})
    grid = totals.merge(cells, how="cross")
    out = grid.merge(events.reset_index(), on=["deal", "arm", "cell"], how="left")
    out["events"] = out.events.fillna(0).astype(int)
    return out[_COUNT_COLUMNS]


def _const(frame: pd.DataFrame, value: str) -> pd.Series:
    return pd.Series(value, index=frame.index)


def _concat(*parts) -> pd.DataFrame:
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else _empty()


def _flag(frame: pd.DataFrame, col: str) -> pd.Series:
    return frame[col].fillna(False).astype(bool)


def _subject(log, kind: str) -> pd.DataFrame:
    d = log.decisions
    return d[(d.kind == kind) & d.is_subject.astype(bool)]


def _subject_plays(log) -> pd.DataFrame:
    p = _subject(log, "play")
    return p[~_flag(p, "forced")]


def _follows(log) -> pd.DataFrame:
    """Non-forced follow Decisions (someone else holds the Trick) with a legal
    beat. With no beat, Pass is the only legal Play — a Forced Play Decision."""
    p = _subject_plays(log)
    return p[p.holder.isin(["partner", "opponent"]) & _flag(p, "has_beat")]


def _seat_rounds(log) -> pd.DataFrame:
    """One row per (Round, Subject seat) — the unit for per-seat outcomes."""
    r = log.rounds
    parts = [r.assign(seat=r.subject_team + offset) for offset in (0, 2)]
    return pd.concat(parts, ignore_index=True)


def _team_values(r: pd.DataFrame, prefix: str):
    """(Subject team's, opponent team's) value of a `<prefix>_0/_1` column pair."""
    mine = r[f"{prefix}_0"].where(r.subject_team == 0, r[f"{prefix}_1"])
    theirs = r[f"{prefix}_1"].where(r.subject_team == 0, r[f"{prefix}_0"])
    return mine, theirs


def _yes_no(series: pd.Series, yes: str, no: str) -> pd.Series:
    return series.fillna(False).astype(bool).map({True: yes, False: no})


# --- A. Round outcome ---------------------------------------------------------

def _slam(log) -> pd.DataFrame:
    r = log.rounds
    slam = r.slam_team
    return _concat(
        _tally(r, _const(r, "for"), slam == r.subject_team),
        _tally(r, _const(r, "against"), slam.notna() & (slam != r.subject_team)),
    )


def _round_points(log) -> pd.DataFrame:
    """Mean Subject-minus-opponent points per Round: in total, and card play alone
    (Call bonuses removed — the split the Tournament Matrix also reports)."""
    r = log.rounds
    mine, theirs = _team_values(r, "total")
    bonus_mine, bonus_theirs = _team_values(r, "call_bonus")
    return _concat(
        _tally_sum(r, _const(r, "total"), mine - theirs),
        _tally_sum(r, _const(r, "card play"), (mine - bonus_mine) - (theirs - bonus_theirs)),
    )


def _out_position(log) -> pd.DataFrame:
    sr = _seat_rounds(log)
    place = [("1st", "2nd", "3rd")[order.index(seat)] if seat in order else "last"
             for seat, order in zip(sr.seat, sr.out_order)]
    return _distribution(sr, pd.Series(place, index=sr.index),
                         categories=("1st", "2nd", "3rd", "last"))


def _trick_share(log) -> pd.DataFrame:
    r = log.rounds
    r = r[r.tricks_total.notna()]          # a replayed human Round has no Trick attribution
    if r.empty:
        return _empty()
    won = [tw[t] + tw[t + 2] for tw, t in zip(r.tricks_won, r.subject_team)]
    t = pd.DataFrame({"deal": r.deal.to_numpy(), "arm": r.arm.to_numpy(), "cell": "all",
                      "events": won, "opportunities": r.tricks_total.to_numpy()})
    return t.groupby(["deal", "arm", "cell"], as_index=False)[["events", "opportunities"]].sum()


# --- B. Calls -----------------------------------------------------------------

def _power_cell(frame: pd.DataFrame) -> pd.Series:
    return frame.power.clip(upper=4).map(lambda p: f"power {p}" + ("+" if p == 4 else ""))


def _call_rate(kind: str):
    def fn(log) -> pd.DataFrame:
        c = _subject(log, kind)
        return _concat(_tally(c, _const(c, "all"), _flag(c, "called")),
                       _tally(c, _power_cell(c), _flag(c, "called")))
    return fn


def _call_success(log) -> pd.DataFrame:
    """Of the Subject's Calls, the fraction whose caller went out first."""
    sr = _seat_rounds(log)
    first = sr.out_order.map(lambda o: o[0] if len(o) else -1)
    parts = []
    for kind, col in (("grand", "grand_callers"), ("tichu", "tichu_callers")):
        called = pd.Series([seat in callers for seat, callers in zip(sr.seat, sr[col])],
                           index=sr.index)
        c = sr[called]
        parts.append(_tally(c, _const(c, kind), first[called] == c.seat))
    return _concat(*parts)


def _call_bonus(log) -> pd.DataFrame:
    r = log.rounds
    mine, _ = _team_values(r, "call_bonus")
    return _tally_sum(r, _const(r, "net"), mine)


def _tichu_call_context(log) -> pd.DataFrame:
    c = _subject(log, "tichu")
    ctx = [("partner called" if p else "opponent called" if o else "nobody called")
           for p, o in zip(_flag(c, "partner_called"), _flag(c, "opponent_called"))]
    return _tally(c, pd.Series(ctx, index=c.index), _flag(c, "called"))


# --- C. Schupfen --------------------------------------------------------------

_DIRECTIONS = (("give_next", "next"), ("give_partner", "partner"), ("give_previous", "previous"))


def _schupfen_card(log) -> pd.DataFrame:
    s = _subject(log, "schupfen")
    return _concat(*(_distribution(s, f"to {direction}: " + s[col]) for col, direction in _DIRECTIONS))


def _special_gift(log) -> pd.DataFrame:
    """For each special card the seat held: kept, given to partner, or to an opponent."""
    s = _subject(log, "schupfen")
    parts = []
    for special in ("dog", "dragon", "phoenix", "mahjong"):
        held = s[_flag(s, f"holds_{special}")]
        where = ["partner" if p == special else "opponent" if special in (n, v) else "kept"
                 for p, n, v in zip(held.give_partner, held.give_next, held.give_previous)]
        parts.append(_distribution(held, special + " → " + pd.Series(where, index=held.index)))
    return _concat(*parts)


def _schupfen_breaks_bomb(log) -> pd.DataFrame:
    s = _subject(log, "schupfen")
    s = s[_flag(s, "had_bomb")]
    return _tally(s, _const(s, "all"), _flag(s, "bomb_broken"))


def _high_card_to_partner(log) -> pd.DataFrame:
    s = _subject(log, "schupfen")
    cell = _yes_no(s.partner_grand, "partner called Grand", "no Grand")
    return _tally(s, cell, s.give_partner.isin(["A", "dragon", "phoenix"]))


# --- D. Play — leading --------------------------------------------------------

def _leads(log) -> pd.DataFrame:
    p = _subject_plays(log)
    return p[p.holder == "none"]


def _lead_type(log) -> pd.DataFrame:
    return _distribution(_leads(log), "action")


def _dog_lead(log) -> pd.DataFrame:
    leads = _leads(log)
    leads = leads[_flag(leads, "holds_dog")]
    cell = _yes_no(leads.partner_called, "partner called", "partner not called")
    return _tally(leads, cell, leads.action == "dog")


# --- E. Play — following ------------------------------------------------------

def _pass_despite_beat(log) -> pd.DataFrame:
    f = _follows(log)
    big = f[f.trick_points >= 10] if "trick_points" in f else f.iloc[0:0]
    return _concat(_tally(f, "holder", f.action == "pass"),
                   _tally(big, big.holder + ", trick ≥10 pts", big.action == "pass"))


def _partner_overtake(log) -> pd.DataFrame:
    f = _follows(log)
    f = f[f.holder == "partner"]
    cell = _yes_no(f.partner_called, "partner called", "partner not called")
    return _tally(f, cell, f.action != "pass")


def _caller_overtake_then_fail(log) -> pd.DataFrame:
    """Round-level 2x2 (per Subject seat): Rounds with a Caller Partner Overtake
    opportunity, split by overtook-at-least-once vs always-yielded; the event is
    the calling partner NOT going out first. Descriptive, not causal — overtakes
    concentrate in Rounds the caller was already losing."""
    f = _follows(log)
    f = f[(f.holder == "partner") & _flag(f, "partner_called")]
    if f.empty:
        return _empty()
    per = (f.assign(overtook=f.action != "pass")
           .groupby(["arm", "deal", "half", "seat"], as_index=False).overtook.any())
    per = per.merge(log.rounds[["arm", "deal", "half", "out_order"]], on=["arm", "deal", "half"])
    failed = [len(o) == 0 or o[0] != (seat + 2) % 4 for seat, o in zip(per.seat, per.out_order)]
    return _tally(per, _yes_no(per.overtook, "overtook", "yielded"),
                  pd.Series(failed, index=per.index))


def _caller_passivity(log) -> pd.DataFrame:
    f = _follows(log)
    f = f[(f.holder == "opponent") & _flag(f, "self_called")]
    bomb = f[_flag(f, "bomb_legal")]
    return _concat(_tally(f, _const(f, "any beat"), f.action == "pass"),
                   _tally(bomb, _const(bomb, "bomb legal"), bomb.action == "pass"))


def _bomb_when_legal(log) -> pd.DataFrame:
    p = _subject_plays(log)
    p = p[_flag(p, "bomb_legal") & p.holder.isin(["partner", "opponent"])]
    return _tally(p, "holder", p.action == "bomb")


def _bombs_per_round(log) -> pd.DataFrame:
    p = _subject(log, "play")
    bombs = p.assign(b=p.action == "bomb").groupby(["arm", "deal", "half"]).b.sum()
    r = log.rounds.set_index(["arm", "deal", "half"])
    r = r.assign(bombs=bombs.reindex(r.index, fill_value=0)).reset_index()
    t = pd.DataFrame({"deal": r.deal, "arm": r.arm, "cell": "per seat-Round",
                      "events": r.bombs, "opportunities": 2})
    return t.groupby(["deal", "arm", "cell"], as_index=False)[["events", "opportunities"]].sum()


def _bomb_unplayed(log) -> pd.DataFrame:
    """Seat-Rounds where the Subject held a Bomb at some Play Decision but never
    played one."""
    p = _subject(log, "play")
    g = (p.assign(held=_flag(p, "holds_bomb"), bombed=p.action == "bomb")
         .groupby(["arm", "deal", "half", "seat"], as_index=False)[["held", "bombed"]].any())
    g = g[g.held]
    return _tally(g, _const(g, "all"), ~g.bombed)


def _phoenix_use(log) -> pd.DataFrame:
    p = _subject(log, "play")
    p = p[_flag(p, "plays_phoenix")]
    how = ["combination" if a != "single" else
           "single lead" if h == "none" else "single over A" if t == 14 else "single other"
           for a, h, t in zip(p.action, p.holder, p.top_rank)]
    return _distribution(p, pd.Series(how, index=p.index))


def _phoenix_combination_when_legal(log) -> pd.DataFrame:
    """Of non-forced Play Decisions where a Phoenix Combination was legal, the
    fraction that played one — the rate the `phoenix_in_combination` lever moves
    (`phoenix_use` is conditioned on playing the Phoenix at all)."""
    p = _subject_plays(log)
    if "phoenix_combo_legal" not in p:     # a log recorded before the field existed
        return _empty()
    p = p[_flag(p, "phoenix_combo_legal")]
    return _tally(p, _const(p, "all"), _flag(p, "plays_phoenix") & (p.action != "single"))


def _dragon_play(log) -> pd.DataFrame:
    p = _subject(log, "play")
    p = p[_flag(p, "plays_dragon")]
    how = ["lead" if h == "none" else
           "follow, trick ≥10 pts" if pts >= 10 else "follow, trick <10 pts"
           for h, pts in zip(p.holder, p.trick_points)]
    return _distribution(p, pd.Series(how, index=p.index))


# --- F. Wish ------------------------------------------------------------------

_WISHES = ("none", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")


def _wish_rank(log) -> pd.DataFrame:
    return _distribution(_subject(log, "wish"), "wish", categories=_WISHES)


def _wish_gave_opponent(log) -> pd.DataFrame:
    """Of the Subject's wishes for a rank, the fraction naming a rank it gave an
    opponent in its own Schupfen (the human heuristic of forcing that card out)."""
    keys = ["arm", "deal", "half", "seat"]
    w = _subject(log, "wish")
    w = w[w.wish != "none"][keys + ["wish"]]
    s = _subject(log, "schupfen")[keys + ["give_next", "give_previous"]]
    w = w.merge(s, on=keys, how="left")
    return _tally(w, _const(w, "all"), (w.wish == w.give_next) | (w.wish == w.give_previous))


def _wish_fulfilled(log) -> pd.DataFrame:
    keys = ["arm", "deal", "half"]
    w = _subject(log, "wish")
    w = w[w.wish != "none"][keys].merge(log.rounds[keys + ["wish_fulfilled"]], on=keys)
    return _tally(w, _const(w, "all"), _flag(w, "wish_fulfilled"))


# --- G. Dragon Assignment -----------------------------------------------------

def _dragon_give(log) -> pd.DataFrame:
    d = _subject(log, "dragon")
    unequal = d[d.target_cards != "equal"] if "target_cards" in d else d
    return _concat(
        _tally(d, _const(d, "to next opponent"), d.target == "next"),
        _tally(unequal, _const(unequal, "to opponent with more cards"), unequal.target_cards == "more"),
        _tally(d, _const(d, "to a calling opponent"), _flag(d, "target_called")),
    )


_A, _B, _C = "A. Round outcome", "B. Calls", "C. Schupfen"
_D, _E, _F, _G = "D. Play — leading", "E. Play — following", "F. Wish", "G. Dragon Assignment"

METRICS: list[Metric] = [
    Metric("slam", _A, "Rounds the Subject team Slams (for) or is Slammed (against).",
           _slam, conditional=False),
    Metric("round_points", _A,
           "Mean Subject-minus-opponent points per Round (total; card play only).",
           _round_points, conditional=False),
    Metric("out_position", _A, "Where each Subject seat finishes (last = never went out).",
           _out_position, conditional=False),
    Metric("trick_share", _A, "Share of the Round's Tricks won by the Subject team.",
           _trick_share, conditional=False),
    Metric("grand_call", _B,
           "Grand-Tichu call rate, overall and by hand power (Dragon + Phoenix + Aces).",
           _call_rate("grand")),
    Metric("tichu_call", _B, "Tichu call rate when asked, overall and by hand power.",
           _call_rate("tichu")),
    Metric("tichu_call_context", _B, "Tichu call rate by who had already called.",
           _tichu_call_context),
    Metric("call_success", _B, "Of the Subject's Calls, the fraction whose caller went out first.",
           _call_success),
    Metric("call_bonus", _B, "Net Call bonus points per Round to the Subject team.",
           _call_bonus, conditional=False),
    Metric("schupfen_card", _C, "Which card goes in each Schupfen direction.", _schupfen_card),
    Metric("special_gift", _C, "Where each held special card goes: kept, partner, opponent.",
           _special_gift),
    Metric("schupfen_breaks_bomb", _C,
           "Of Schupfens holding a Bomb, the fraction giving a Bomb card away.",
           _schupfen_breaks_bomb),
    Metric("high_card_to_partner", _C,
           "Ace / Dragon / Phoenix given to partner, by whether partner called Grand.",
           _high_card_to_partner),
    Metric("lead_type", _D, "Combination type when leading.", _lead_type),
    Metric("dog_lead", _D, "Leads with the Dog when holding it, by whether partner called.",
           _dog_lead),
    Metric("pass_despite_beat", _E, "Pass rate when a legal beat exists, by who holds the Trick.",
           _pass_despite_beat),
    Metric("partner_overtake", _E,
           "Partner Overtake rate when partner holds the Trick and a beat exists.",
           _partner_overtake),
    Metric("caller_overtake_then_fail", _E,
           "Rounds with a Caller Partner Overtake chance: partner's call failed, "
           "by overtook vs always yielded (descriptive, not causal).",
           _caller_overtake_then_fail),
    Metric("caller_passivity", _E, "As a caller following an opponent with a beat, the Pass rate.",
           _caller_passivity),
    Metric("bomb_when_legal", _E, "Bombs when a Bomb is legal, by who holds the Trick.",
           _bomb_when_legal),
    Metric("bombs_per_round", _E, "Bombs played per Subject seat-Round.", _bombs_per_round,
           conditional=False),
    Metric("bomb_unplayed", _E, "Seat-Rounds holding a Bomb that never play one.", _bomb_unplayed),
    Metric("phoenix_use", _E, "How the Phoenix is played.", _phoenix_use),
    Metric("phoenix_combination_when_legal", _E,
           "Plays a Phoenix Combination when one is legal.", _phoenix_combination_when_legal),
    Metric("dragon_play", _E, "When the Dragon is played.", _dragon_play),
    Metric("wish_rank", _F, "Wished rank (none = declined).", _wish_rank),
    Metric("wish_gave_opponent", _F,
           "Wishes naming a rank the wisher gave an opponent in Schupfen.", _wish_gave_opponent),
    Metric("wish_fulfilled", _F, "Wishes fulfilled before the Round ends.", _wish_fulfilled),
    Metric("dragon_give", _G, "Who receives a Dragon Trick.", _dragon_give),
]


def metric(name: str) -> Metric:
    return next(m for m in METRICS if m.name == name)


def summarise(log, metrics=None, *, n_boot: int = 2000, seed: int = 0,
              floor: int = 200, q: float = 0.05, references=()) -> pd.DataFrame:
    """The tidy panel: every metric's cells x {bc, subject, delta} — plus the
    Synthetic-Game metrics (`metric == "game"`) — with `bh_survives` set on Δ rows
    that survive Benjamini–Hochberg at FDR `q` across the whole panel.

    `references` are unpaired reference logs (the human games, `drift_human`):
    each adds rows under its own `arm` name with a Game-clustered CI and no Δ.
    They never enter the BH family — that is the Subject-vs-BC comparison only."""
    deals = sorted(set(log.rounds.deal))
    parts = []
    for m in metrics or METRICS:
        s = summarise_cells(m.fn(log), deals=deals, n_boot=n_boot, seed=seed, floor=floor)
        s.insert(0, "family", m.family)
        s.insert(0, "metric", m.name)
        s["conditional"] = m.conditional
        parts.append(s)
    games = summarise_games(log.rounds, n_boot=n_boot, seed=seed)
    games.insert(0, "family", "A. Round outcome")
    games.insert(0, "metric", "game")
    games["conditional"] = False
    parts.append(games)
    panel = pd.concat(parts, ignore_index=True)
    is_delta = (panel.arm == "delta").to_numpy()
    panel["bh_survives"] = False
    panel.loc[is_delta, "bh_survives"] = benjamini_hochberg(panel.p[is_delta], q=q)
    refs = [_summarise_reference(ref, metrics or METRICS, n_boot=n_boot, seed=seed, floor=floor)
            for ref in references]
    return pd.concat([panel, *refs], ignore_index=True) if refs else panel


def _summarise_reference(ref, metrics, *, n_boot, seed, floor) -> pd.DataFrame:
    parts = []
    for m in metrics:
        s = summarise_reference(m.fn(ref), ref.rounds, n_boot=n_boot, seed=seed, floor=floor)
        s.insert(0, "family", m.family)
        s.insert(0, "metric", m.name)
        s["conditional"] = m.conditional
        parts.append(s)
    games = reference_games(ref.rounds, n_boot=n_boot, seed=seed)
    games.insert(0, "family", "A. Round outcome")
    games.insert(0, "metric", "game")
    games["conditional"] = False
    parts.append(games)
    return pd.concat(parts, ignore_index=True)
