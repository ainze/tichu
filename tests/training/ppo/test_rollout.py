"""PPO Refine self-play rollout harness (ADR-0029).

The driver runs self-play Rounds under an injected play-policy and emits one
trajectory per learner-team seat, each carrying the terminal `round_outcome`
reward (team-relative, call-bonus-inclusive — see `_finalise_round`).
"""

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent

from tichu_training.ppo.rollout import (
    CallChoice,
    PlayChoice,
    SchupfenChoice,
    collect_rollout,
)


class RulePlayPolicy:
    """A play-policy that defers each Play decision to a RuleAgent.

    A rollout driven by this policy makes the same moves a `play_full_round` of
    RuleAgents makes, so the two must reach the same Round outcome — the
    cross-check that proves the driver reproduces real engine dynamics.
    """

    def __init__(self) -> None:
        self._rule = RuleAgent()

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        return [
            PlayChoice(concrete_action=self._rule.act(private_state))
            for _seat, private_state, _gs in decisions
        ]


class TaggingPolicy:
    """Plays RuleAgent moves but stamps each PlayChoice with a globally-unique
    id (and mirrors it into logprob/value), recording per seat what it returned.
    Globally-unique tags mean any mis-ordering or opponent-leak into a learner
    trajectory fails the match.
    """

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.returned: dict[int, list[int]] = {}
        self._n = 0

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        out = []
        for seat, private_state, _gs in decisions:
            tag = self._n
            self._n += 1
            self.returned.setdefault(seat, []).append(tag)
            out.append(
                PlayChoice(
                    concrete_action=self._rule.act(private_state),
                    intent_index=tag,
                    logprob=float(-tag),
                    value=float(tag),
                )
            )
        return out


def test_rollout_reward_matches_trusted_runner_outcome():
    pos = generate_full_position_pool(seed=0, n=1)[0]

    # Reference outcome from the established Full-strength runner: RuleAgents,
    # no calls (RuleAgent exposes no `should_call`).
    ref = play_full_round(
        tuple(RuleAgent() for _ in range(4)), pos.state, pos.grand_prefixes
    )
    expected_team0_relative = ref.total[0] - ref.total[1]

    trajs = collect_rollout([pos], RulePlayPolicy(), learner_team=0)

    # One trajectory per learner-team seat: team 0 -> seats 0 and 2.
    assert {t.seat for t in trajs} == {0, 2}
    # Terminal reward is the team-relative round_outcome.
    assert all(t.reward == expected_team0_relative for t in trajs)


class BatchSpyPolicy:
    """Plays RuleAgent moves and records the size of every batch it is asked to
    decide — so a test can prove the driver batches Play decisions across games."""

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.batch_sizes: list[int] = []

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        self.batch_sizes.append(len(decisions))
        return [
            PlayChoice(concrete_action=self._rule.act(ps)) for _seat, ps, _gs in decisions
        ]


class SeatRecordingPolicy:
    """Plays RuleAgent moves and records which seats it was asked to decide for."""

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.seats_seen: list[int] = []

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        self.seats_seen += [seat for seat, _, _ in decisions]
        return [
            PlayChoice(concrete_action=self._rule.act(ps)) for _seat, ps, _gs in decisions
        ]


def test_opponent_seats_are_routed_to_the_opponent_policy():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    learner = SeatRecordingPolicy()
    opponent = SeatRecordingPolicy()

    trajs = collect_rollout(
        [pos], learner, opponent_policy=opponent, learner_team=0
    )

    # The learner policy decides only its own team's seats; opponents the rest.
    assert set(learner.seats_seen) == {0, 2}
    assert set(opponent.seats_seen) == {1, 3}
    # Trajectories are still recorded only for the learner team.
    assert {t.seat for t in trajs} == {0, 2}


def _signature(trajs):
    """Order-independent fingerprint of a rollout's learner trajectories."""
    return sorted((t.seat, t.reward, len(t.steps)) for t in trajs)


def test_vectorized_rollout_matches_serial_and_batches_across_games():
    positions = generate_full_position_pool(seed=0, n=3)

    # Trusted reference: each game rolled out on its own.
    serial = []
    for pos in positions:
        serial += collect_rollout([pos], BatchSpyPolicy(), learner_team=0)

    spy = BatchSpyPolicy()
    batched = collect_rollout(positions, spy, learner_team=0)

    # (a) Invariance: interleaving many games corrupts no per-game state.
    assert _signature(batched) == _signature(serial)
    # (b) Real batching: at least one tick decided Play for >1 game at once.
    assert max(spy.batch_sizes) > 1


class ScriptedCaller(RuleAgent):
    """RuleAgent (identical, deterministic play) that answers Call Decisions on
    command — lets a test hold Play fixed while toggling a call."""

    def __init__(self, *, grand: bool = False, tichu: bool = False) -> None:
        super().__init__()
        self._grand = grand
        self._tichu = tichu

    def should_call(self, private_state, kind: str) -> bool:
        return self._grand if kind == "grand" else self._tichu


def test_terminal_reward_includes_grand_tichu_call_bonus():
    pos = generate_full_position_pool(seed=3, n=1)[0]

    # Identical RuleAgent play in both runs (Play comes from the policy); the
    # only difference is seat 0 (learner team) calling Grand-Tichu.
    calling = [ScriptedCaller(grand=True), RuleAgent(), RuleAgent(), RuleAgent()]
    plain = [RuleAgent() for _ in range(4)]

    r_call = collect_rollout([pos], RulePlayPolicy(), learner_team=0, seat_agents=calling)
    r_plain = collect_rollout([pos], RulePlayPolicy(), learner_team=0, seat_agents=plain)

    reward_call = next(t.reward for t in r_call if t.seat == 0)
    reward_plain = next(t.reward for t in r_plain if t.seat == 0)

    # A Grand-Tichu by a team-0 seat shifts the team-relative outcome by exactly
    # +/-200 (made or busted); nothing else changes.
    assert abs(reward_call - reward_plain) == 200


def test_terminal_reward_includes_tichu_call_bonus():
    pos = generate_full_position_pool(seed=3, n=1)[0]

    # Identical RuleAgent play in both runs (Play comes from the policy); the only
    # difference is seat 0 (learner team) calling regular Tichu — solicited at its
    # first non-Pass Play (ADR-0018), the timing `play_full_round` uses. The driver
    # used to drop Tichu entirely (ADR-0029 froze calls); the full-stack collector
    # must solicit it so the +/-100 bonus lands in `round_outcome` (ADR-0034).
    calling = [ScriptedCaller(tichu=True), RuleAgent(), RuleAgent(), RuleAgent()]
    plain = [RuleAgent() for _ in range(4)]

    r_call = collect_rollout([pos], RulePlayPolicy(), learner_team=0, seat_agents=calling)
    r_plain = collect_rollout([pos], RulePlayPolicy(), learner_team=0, seat_agents=plain)

    reward_call = next(t.reward for t in r_call if t.seat == 0)
    reward_plain = next(t.reward for t in r_plain if t.seat == 0)

    # A Tichu by a team-0 seat shifts the team-relative outcome by exactly +/-100
    # (made or busted); play is identical, so nothing else moves.
    assert abs(reward_call - reward_plain) == 100


class SchupfenTaggingPolicy:
    """Plays RuleAgent moves AND serves Schupfen (via RuleAgent) but stamps each
    schupfen with a globally-unique tag, recording which seats it was asked — so a
    test can prove learner Schupfen is routed to the policy and recorded once per
    learner seat, tagged, with the bookkeeping the policy returned (ADR-0034)."""

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.schupfen_seats: list[int] = []
        self._n = 1000

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        return [PlayChoice(concrete_action=self._rule.act(ps)) for _s, ps, _gs in decisions]

    def act_schupfen_batch(self, decisions) -> list[SchupfenChoice]:
        out = []
        for seat, ps, _gs in decisions:
            tag = self._n
            self._n += 1
            self.schupfen_seats.append(seat)
            out.append(
                SchupfenChoice(
                    concrete_action=self._rule.act(ps),
                    card_indices=(tag, tag, tag),
                    logprob=float(-tag),
                    value=float(tag),
                )
            )
        return out


def test_records_learner_schupfen_when_policy_supports_it():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    learner = SchupfenTaggingPolicy()
    # Opponent has no act_schupfen_batch -> opponent Schupfen stays on the frozen
    # inline seat agent, so only learner seats reach the learner's schupfen method.
    trajs = collect_rollout(
        [pos], learner, opponent_policy=RulePlayPolicy(), learner_team=0
    )
    by_seat = {t.seat: t for t in trajs}

    for seat in (0, 2):
        sch = [s for s in by_seat[seat].steps if s.decision_type == "schupfen"]
        assert len(sch) == 1, f"seat {seat} should schupfen exactly once"
        # Recorded with exactly the bookkeeping the policy returned.
        assert sch[0].logprob == float(-sch[0].intent_index[0])
        assert sch[0].value == float(sch[0].intent_index[0])
    # Only learner-team seats were routed to the learner's schupfen method.
    assert set(learner.schupfen_seats) == {0, 2}


class CallTaggingPolicy:
    """Plays RuleAgent moves and serves Calls via `act_call_batch`, scripting seat 0
    to call Grand-Tichu and seat 2 to call Tichu, stamping each solicitation with a
    unique tag. Records every call decision (both call and skip are policy actions to
    train on), so a test can prove learner calls are routed + recorded with the right
    `decision_type`, and that a Grand-Tichu caller is never asked Tichu (ADR-0034)."""

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.solicited: list[tuple[str, int]] = []
        self._n = 5000

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        return [PlayChoice(concrete_action=self._rule.act(ps)) for _s, ps, _gs in decisions]

    def act_call_batch(self, kind: str, decisions) -> list[CallChoice]:
        out = []
        for seat, _ps, _gs in decisions:
            tag = self._n
            self._n += 1
            self.solicited.append((kind, seat))
            called = (kind == "grand" and seat == 0) or (kind == "tichu" and seat == 2)
            out.append(CallChoice(called=called, logprob=float(-tag), value=float(tag)))
        return out


def test_records_learner_calls_and_grand_supersedes_tichu():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    learner = CallTaggingPolicy()
    trajs = collect_rollout(
        [pos], learner, opponent_policy=RulePlayPolicy(), learner_team=0
    )
    by_seat = {t.seat: t for t in trajs}

    def kinds(seat):
        return [s.decision_type for s in by_seat[seat].steps if s.decision_type in ("grand", "tichu")]

    # Every learner seat is asked Grand at deal time. Seat 0 called Grand, so it is
    # NOT asked Tichu; seat 2 declined Grand, so it IS asked Tichu at first non-Pass.
    assert kinds(0) == ["grand"]
    assert kinds(2) == ["grand", "tichu"]
    # Only learner seats were routed to the learner's call method.
    assert {seat for _kind, seat in learner.solicited} == {0, 2}
    # Recorded with the bookkeeping the policy returned (logprob == -tag).
    for seat in (0, 2):
        for s in by_seat[seat].steps:
            if s.decision_type in ("grand", "tichu"):
                assert s.logprob == float(-s.value)


def test_outcome_matches_trusted_runner_with_a_tichu_caller():
    # ADR-0034 parity gate: the full-stack collector, driven by the same agents as
    # the trusted `play_full_round`, must reach the same Round outcome even with a
    # caller in play — so it inherits the runner's validated call/play dynamics.
    pos = generate_full_position_pool(seed=0, n=1)[0]
    agents = [ScriptedCaller(tichu=True), RuleAgent(), RuleAgent(), RuleAgent()]

    ref = play_full_round(tuple(agents), pos.state, pos.grand_prefixes)
    expected_team0_relative = ref.total[0] - ref.total[1]

    trajs = collect_rollout(
        [pos], RulePlayPolicy(), learner_team=0, seat_agents=agents
    )
    assert all(t.reward == expected_team0_relative for t in trajs)


# --- Terminal-reward parity on scenarios RuleAgent self-play almost never reaches.
# The collector has no slam/last-hand-specific code: it steps the engine and reads
# the terminal score delta. These crafted near-end states drive the collector to a
# slam / phoenix-in-last-hand finish and assert its reward equals the engine oracle
# (a direct `step` of the same state), exercising the paths the seed-swept parity
# tests above cannot hit. (Audit: co-train scoring, 2026-06-07.)

from tichu_eval.full_position_pool import FullStartingPosition  # noqa: E402
from tichu_engine.cards import Card, PHOENIX, Suit  # noqa: E402
from tichu_engine.combinations import Single  # noqa: E402
from tichu_engine.engine import step  # noqa: E402
from tichu_engine.state import GameState, PublicState, Trick  # noqa: E402


def _card(rank, suit=Suit.STAR):
    return Card(suit=suit, rank=rank)


def _collector_reward(state):
    pos = FullStartingPosition(state=state, grand_prefixes=(frozenset(),) * 4)
    trajs = collect_rollout([pos], RulePlayPolicy(), learner_team=0)
    return next(t.reward for t in trajs if t.seat == 0)


def _oracle_reward(state, action):
    init = state.public.scores
    ns, _, done, _ = step(state, action)
    assert done
    f = ns.public.scores
    return (f[0] - init[0]) - (f[1] - init[1]), ns.public.scores


def test_collector_terminal_reward_matches_engine_oracle_on_slam():
    # Seat 0 (team 0) already out; partner seat 2 plays its last (0-point) card ->
    # Doppelsieg. Collector's team-relative reward must equal the engine oracle and
    # the absolute finalise must be exactly the +200 slam bonus, no card-point leak.
    state = GameState(
        hands=(frozenset(), frozenset({_card(9)}),
               frozenset({_card(9, Suit.PAGODA)}), frozenset({_card(4, Suit.SWORD)})),
        public=PublicState(current_player=2, hand_sizes=(0, 1, 1, 1),
                           scores=(0, 0), trick=Trick.empty(), out_order=(0,)),
    )
    oracle, final = _oracle_reward(state, Single(_card(9, Suit.PAGODA)))
    assert final == (200, 0)
    assert _collector_reward(state) == oracle == 200


def test_collector_terminal_reward_matches_engine_oracle_on_phoenix_last_hand():
    # Non-slam finish: seats 0,1 out; seat 2 (team 0) plays its last card to go out
    # third; seat 3 (team 1) is last-in holding the Phoenix. Phoenix's -25 transfers
    # to the OPPOSING team (team 0). Collector reward must equal the engine oracle.
    state = GameState(
        hands=(frozenset(), frozenset(), frozenset({_card(9)}), frozenset({PHOENIX})),
        public=PublicState(current_player=2, hand_sizes=(0, 0, 1, 1),
                           scores=(0, 0), trick=Trick.empty(), out_order=(0, 1)),
    )
    oracle, _ = _oracle_reward(state, Single(_card(9)))
    assert _collector_reward(state) == oracle


def test_records_each_learner_play_decision_in_order_and_no_opponent_leak():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    policy = TaggingPolicy()

    trajs = collect_rollout([pos], policy, learner_team=0)
    by_seat = {t.seat: t for t in trajs}

    # Only learner-team seats get trajectories — opponents (1, 3) get none.
    assert set(by_seat) == {0, 2}
    for seat in (0, 2):
        steps = by_seat[seat].steps
        # The learner acted, and every Play decision it made is recorded once,
        # in order, with exactly the bookkeeping the policy returned.
        assert [s.intent_index for s in steps] == policy.returned.get(seat, [])
        assert len(steps) > 0
        for s in steps:
            assert s.logprob == float(-s.intent_index)
            assert s.value == float(s.intent_index)
