"""PPO Refine self-play rollout harness (ADR-0029).

The driver runs self-play Rounds under an injected play-policy and emits one
trajectory per learner-team seat, each carrying the terminal `round_outcome`
reward (team-relative, call-bonus-inclusive — see `_finalise_round`).
"""

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent

from tichu_training.ppo.rollout import PlayChoice, collect_rollout


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
            for _seat, private_state in decisions
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
        for seat, private_state in decisions:
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
            PlayChoice(concrete_action=self._rule.act(ps)) for _seat, ps in decisions
        ]


class SeatRecordingPolicy:
    """Plays RuleAgent moves and records which seats it was asked to decide for."""

    def __init__(self) -> None:
        self._rule = RuleAgent()
        self.seats_seen: list[int] = []

    def act_play_batch(self, decisions) -> list[PlayChoice]:
        self.seats_seen += [seat for seat, _ in decisions]
        return [
            PlayChoice(concrete_action=self._rule.act(ps)) for _seat, ps in decisions
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
