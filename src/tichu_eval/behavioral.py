"""Behavioral telemetry profile for an Agent.

Answers "*how* does this Agent play?" rather than "*how strong* is it?" — the
Tournament Matrix measures strength; this measures behavior. Built first as the
diagnostic instrument for the soft-policy / never-bombs hypothesis (does the
`master` ML Agent ever Bomb? call Tichu? win its share of Tricks?), and reused
later as the PPO Refine tuning dashboard (online RL is steered by watching
behavior drift, not win-rate alone). See CONTEXT.md §"Phase 2 / online-RL terms".

Each Agent is profiled in self-play: seated in all four seats of a Full-strength
Round over the Starting-Position Pool, so the record is its behavior at a
homogeneous table. Per-seat `RoundTelemetry` (emitted by `play_full_round(...,
collect_telemetry=True)`) is folded into per-Agent counters; rates are derived.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tichu_eval.play_full import play_full_round


def _team_of(seat: int) -> int:
    return seat % 2


@dataclass
class BehavioralProfile:
    """Accumulated behavioral counters for one Agent, plus derived rates.

    A "seat-round" is one seat's participation in one Round — the natural unit,
    since an Agent occupies multiple seats. Neutral references: with 4 equal
    seats, `trick_win_rate` and `out_first_rate` are 0.25 at parity.
    """

    seat_rounds: int = 0
    grand_called: int = 0
    grand_success: int = 0
    tichu_called: int = 0
    tichu_success: int = 0
    bombs_played: int = 0
    bomb_legal_decisions: int = 0
    tricks_won: int = 0
    tricks_available: int = 0
    out_first: int = 0
    slam_for: int = 0
    caller_pass_opportunities: int = 0
    caller_pass_events: int = 0
    caller_pass_bomb_opportunities: int = 0
    caller_pass_bomb_events: int = 0
    partner_steal_opportunities: int = 0
    partner_steal_events: int = 0
    partner_steal_caller_opportunities: int = 0
    partner_steal_caller_events: int = 0

    def _safe(self, num: int, den: int) -> float:
        return num / den if den else 0.0

    @property
    def bomb_per_round(self) -> float:
        return self._safe(self.bombs_played, self.seat_rounds)

    @property
    def bomb_when_legal_rate(self) -> float:
        """Of the Play decisions where a Bomb was legal, the fraction that
        actually Bombed. The headline soft-policy diagnostic."""
        return self._safe(self.bombs_played, self.bomb_legal_decisions)

    @property
    def grand_call_rate(self) -> float:
        return self._safe(self.grand_called, self.seat_rounds)

    @property
    def grand_success_rate(self) -> float:
        return self._safe(self.grand_success, self.grand_called)

    @property
    def tichu_call_rate(self) -> float:
        return self._safe(self.tichu_called, self.seat_rounds)

    @property
    def tichu_success_rate(self) -> float:
        return self._safe(self.tichu_success, self.tichu_called)

    @property
    def trick_win_rate(self) -> float:
        """Tricks won / tricks resolved (neutral 0.25 at a 4-seat parity table)."""
        return self._safe(self.tricks_won, self.tricks_available)

    @property
    def out_first_rate(self) -> float:
        """Fraction of seat-rounds this seat went out first (neutral 0.25)."""
        return self._safe(self.out_first, self.seat_rounds)

    @property
    def slam_rate(self) -> float:
        """Fraction of seat-rounds whose team scored a Doppelsieg (slam)."""
        return self._safe(self.slam_for, self.seat_rounds)

    @property
    def caller_passivity_rate(self) -> float:
        """As a Tichu/Grand caller following an opponent with a legal beating
        play, the fraction of those tricks it ceded by Passing. High = a caller
        that fails to press its lead (the decision-tape-flagged weakness)."""
        return self._safe(self.caller_pass_events, self.caller_pass_opportunities)

    @property
    def caller_bomb_passivity_rate(self) -> float:
        """The unambiguous subset: as a caller with a legal BOMB and an opponent
        winning, the fraction of those tricks it ceded by Passing. There is almost
        never a reason to do this — a near-pure error rate."""
        return self._safe(self.caller_pass_bomb_events, self.caller_pass_bomb_opportunities)

    @property
    def partner_steal_rate(self) -> float:
        """When the partner currently holds the Trick and a legal beat is
        available, the fraction of those the seat OVERTOOK instead of ceding to
        let the partner win. High = stealing partner tricks."""
        return self._safe(self.partner_steal_events, self.partner_steal_opportunities)

    @property
    def partner_steal_caller_rate(self) -> float:
        """The near-pure blunder subset: partner is a Tichu/Grand caller (wants to
        win the Trick / go out), yet the seat overtook it. The metric that targets
        the observed 'bombs/overtakes a calling, winning partner' pathology."""
        return self._safe(self.partner_steal_caller_events, self.partner_steal_caller_opportunities)

    def as_row(self) -> dict[str, float | int]:
        return {
            "seat_rounds": self.seat_rounds,
            "bomb_per_round": round(self.bomb_per_round, 5),
            "bomb_when_legal_rate": round(self.bomb_when_legal_rate, 5),
            "bombs_played": self.bombs_played,
            "bomb_legal_decisions": self.bomb_legal_decisions,
            "grand_call_rate": round(self.grand_call_rate, 5),
            "grand_success_rate": round(self.grand_success_rate, 5),
            "tichu_call_rate": round(self.tichu_call_rate, 5),
            "tichu_success_rate": round(self.tichu_success_rate, 5),
            "trick_win_rate": round(self.trick_win_rate, 5),
            "out_first_rate": round(self.out_first_rate, 5),
            "slam_rate": round(self.slam_rate, 5),
            "caller_passivity_rate": round(self.caller_passivity_rate, 5),
            "caller_pass_opportunities": self.caller_pass_opportunities,
            "caller_bomb_passivity_rate": round(self.caller_bomb_passivity_rate, 5),
            "caller_pass_bomb_opportunities": self.caller_pass_bomb_opportunities,
            "partner_steal_rate": round(self.partner_steal_rate, 5),
            "partner_steal_opportunities": self.partner_steal_opportunities,
            "partner_steal_caller_rate": round(self.partner_steal_caller_rate, 5),
            "partner_steal_caller_opportunities": self.partner_steal_caller_opportunities,
        }


def _fold_round(profile: BehavioralProfile, tel, seat: int) -> None:
    """Fold one seat's slice of one Round's telemetry into the profile."""
    profile.seat_rounds += 1
    profile.grand_called += int(tel.grand_called[seat])
    profile.tichu_called += int(tel.tichu_called[seat])
    is_first_out = tel.first_out == seat
    profile.grand_success += int(tel.grand_called[seat] and is_first_out)
    profile.tichu_success += int(tel.tichu_called[seat] and is_first_out)
    profile.bombs_played += tel.bombs_played[seat]
    profile.bomb_legal_decisions += tel.bomb_legal_decisions[seat]
    profile.tricks_won += tel.tricks_won[seat]
    profile.tricks_available += tel.tricks_total
    profile.out_first += int(is_first_out)
    profile.caller_pass_opportunities += tel.caller_pass_opportunities[seat]
    profile.caller_pass_events += tel.caller_pass_events[seat]
    profile.caller_pass_bomb_opportunities += tel.caller_pass_bomb_opportunities[seat]
    profile.caller_pass_bomb_events += tel.caller_pass_bomb_events[seat]
    profile.partner_steal_opportunities += tel.partner_steal_opportunities[seat]
    profile.partner_steal_events += tel.partner_steal_events[seat]
    profile.partner_steal_caller_opportunities += tel.partner_steal_caller_opportunities[seat]
    profile.partner_steal_caller_events += tel.partner_steal_caller_events[seat]
    # Doppelsieg: the first two players out are partners; both seats of the
    # winning team are credited.
    if (
        len(tel.out_order) >= 2
        and _team_of(tel.out_order[0]) == _team_of(tel.out_order[1])
        and _team_of(seat) == _team_of(tel.out_order[0])
    ):
        profile.slam_for += 1


def profile_agent_self_play(agent_builder, positions) -> BehavioralProfile:
    """Profile one Agent seated in all four seats over the Pool.

    `agent_builder` is a zero-arg callable (same contract as the Tournament): a
    fresh instance per seat, so stateful agents do not bleed across seats.
    """
    profile = BehavioralProfile()
    for pos in positions:
        agents = tuple(agent_builder() for _ in range(4))
        result = play_full_round(
            agents, pos.state, pos.grand_prefixes, collect_telemetry=True
        )
        tel = result.telemetry
        assert tel is not None  # collect_telemetry=True guarantees it
        for seat in range(4):
            _fold_round(profile, tel, seat)
    return profile


def run_behavioral_profiles(
    agent_builders: dict, positions
) -> dict[str, BehavioralProfile]:
    """Profile every named Agent independently in self-play over the Pool."""
    return {
        name: profile_agent_self_play(builder, positions)
        for name, builder in agent_builders.items()
    }
