"""Self-play rollout driver for PPO Refine (ADR-0029).

Drives self-play Rounds through the rules engine under an injected play-policy
and emits one `Trajectory` per learner-team seat, each carrying the terminal
`round_outcome` reward (team-relative, call-bonus-inclusive — the bonus is
folded into `scores` by `_finalise_round` when callers are set on the state).

Only the *Play* Decision is routed to the learning policy; the rare non-Play
Decisions (Schupfen / Wish / Dragon) are handled inline by a frozen baseline
agent, and Calls are out of PPO scope. The driver records steps only for the
two learner-team seats (parameter-shared partner: both are on-policy for the
shared policy; opponents are environment dynamics).

This module is single-game for now; vectorized many-game stepping is a later
slice that preserves this public interface.
"""

from dataclasses import dataclass, field, replace
from typing import NamedTuple, Protocol, Sequence

from tichu_engine.engine import step
from tichu_engine.legality import ConcreteAction
from tichu_engine.state import GameState, PublicState, Trick
from tichu_ml.agent import Agent
from tichu_ml.rule_agent import RuleAgent

_NUM_PLAYERS = 4
_MAX_STEPS = 10_000  # safety net against an infinite-loop bug.


class PlayChoice(NamedTuple):
    """A play-policy's decision at one Play Decision.

    `concrete_action` is what the engine steps. The remaining fields are the
    PPO bookkeeping the learner records for its own seats; a non-learning policy
    (e.g. a frozen opponent or a deterministic fake) leaves them `None`.
    """

    concrete_action: ConcreteAction
    intent_index: int | None = None
    logprob: float | None = None
    value: float | None = None


class RolloutPolicy(Protocol):
    """The play-policy injected into the driver.

    `act_play_batch` receives all Play Decisions live across the concurrent
    games at one tick — a list of `(seat, private_state)` — and returns one
    `PlayChoice` per decision, in the same order. Batching the forward over many
    games at once is the throughput design of ADR-0029.
    """

    def act_play_batch(
        self, decisions: list[tuple[int, object]]
    ) -> list[PlayChoice]: ...


class TrajectoryStep(NamedTuple):
    intent_index: int | None
    logprob: float | None
    value: float | None


@dataclass
class Trajectory:
    """One learner-team seat's recorded experience for a single Round."""

    seat: int
    team: int
    reward: float = 0.0
    steps: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class _PlayGroup:
    """One serving model's batched Play Decisions for a single tick."""

    batch: list[tuple[int, object]] = field(default_factory=list)  # (seat, private_state)
    meta: list = field(default_factory=list)                       # (run, seat) parallel to batch


@dataclass
class _GameRun:
    """Mutable per-game state threaded through the vectorized drive loop."""

    state: GameState
    seat_agents: Sequence[Agent]
    learner_team: int
    initial_scores: tuple[int, int]
    trajs: dict[int, Trajectory]
    done: bool = False


def collect_rollout(
    positions: Sequence,
    policy: RolloutPolicy,
    *,
    opponent_policy: RolloutPolicy | None = None,
    learner_team: int = 0,
    seat_agents: Sequence[Agent] | None = None,
) -> list[Trajectory]:
    """Roll out the Starting Positions concurrently under `policy`, returning one
    Trajectory per learner-team seat per Round.

    The games advance in lockstep: at each tick every live game contributes its
    current Decision. Play Decisions are split by team and batched into one
    `act_play_batch` call per serving model — the learner-team seats go to
    `policy`, the opposing seats to `opponent_policy` (the league / frozen-BC
    opponent; defaults to `policy` for pure self-play). The rare non-Play
    Decisions (Schupfen / Wish / Dragon) plus Calls are served inline by the
    four frozen per-seat `seat_agents` (default: four RuleAgents, which decline
    every call).
    """
    if opponent_policy is None:
        opponent_policy = policy
    if seat_agents is None:
        seat_agents = [RuleAgent() for _ in range(_NUM_PLAYERS)]
    if len(seat_agents) != _NUM_PLAYERS:
        raise ValueError(f"expected {_NUM_PLAYERS} seat_agents, got {len(seat_agents)}")
    learner_seats = [s for s in range(_NUM_PLAYERS) if s % 2 == learner_team]

    runs: list[_GameRun] = []
    for pos in positions:
        grand_callers = _ask_grand(seat_agents, pos.grand_prefixes)
        runs.append(
            _GameRun(
                state=_inject_callers(pos.state, grand=grand_callers),
                seat_agents=seat_agents,
                learner_team=learner_team,
                initial_scores=pos.state.public.scores,
                trajs={
                    seat: Trajectory(seat=seat, team=learner_team)
                    for seat in learner_seats
                },
            )
        )

    _drive(runs, policy, opponent_policy, learner_team)

    out: list[Trajectory] = []
    for run in runs:
        out.extend(run.trajs.values())
    return out


def _drive(
    runs: list[_GameRun],
    policy: RolloutPolicy,
    opponent_policy: RolloutPolicy,
    learner_team: int,
) -> None:
    # Each live game advances exactly one Decision per tick, so the longest game
    # bounds the tick count.
    for _ in range(_MAX_STEPS):
        if all(run.done for run in runs):
            break
        # Play Decisions are grouped by serving model so each model runs a single
        # batched forward per tick (ADR-0029). learner-team seats -> `policy`,
        # opposing seats -> `opponent_policy`.
        groups: dict[RolloutPolicy, _PlayGroup] = {
            policy: _PlayGroup(), opponent_policy: _PlayGroup(),
        }
        for run in runs:
            if run.done:
                continue
            current = run.state.public.current_player
            private = run.state.private_view(current)
            if run.state.public.pending_decision is None:  # a Play Decision
                model = policy if current % 2 == learner_team else opponent_policy
                group = groups[model]
                group.meta.append((run, current))
                group.batch.append((current, private))
            else:  # Schupfen / Wish / Dragon — frozen per-seat agent, inline
                _advance(run, run.seat_agents[current].act(private))
        for model, group in groups.items():
            if not group.batch:
                continue
            choices = model.act_play_batch(group.batch)
            for (run, seat), choice in zip(group.meta, choices):
                if seat in run.trajs:
                    run.trajs[seat].steps.append(
                        TrajectoryStep(choice.intent_index, choice.logprob, choice.value)
                    )
                _advance(run, choice.concrete_action)
    else:
        raise RuntimeError(
            f"rollout exceeded {_MAX_STEPS} ticks without resolving — "
            "likely an infinite loop in policy or engine."
        )

    for run in runs:
        final = run.state.public.scores
        other = 1 - run.learner_team
        reward = float(
            (final[run.learner_team] - run.initial_scores[run.learner_team])
            - (final[other] - run.initial_scores[other])
        )
        for traj in run.trajs.values():
            traj.reward = reward


def _advance(run: _GameRun, action: ConcreteAction) -> None:
    run.state, _, done, _ = step(run.state, action)
    if done:
        run.done = True


def _ask_grand(seat_agents: Sequence[Agent], grand_prefixes: Sequence) -> frozenset[int]:
    """Ask each seat for a Grand-Tichu Call on its synthetic (8,8,8,8) deal-time
    state, independently — matching how the Grand-Tichu networks were trained
    (no prior callers visible). Mirrors `tichu_eval.play_full._ask_grand`."""
    hands = tuple(frozenset(p) for p in grand_prefixes)
    grand_state = GameState(
        hands=hands,
        public=PublicState(
            current_player=0,
            hand_sizes=tuple(len(h) for h in hands),
            scores=(0, 0),
            trick=Trick.empty(),
        ),
    )
    callers = {
        seat
        for seat in range(_NUM_PLAYERS)
        if _calls(seat_agents[seat], grand_state.private_view(seat), "grand")
    }
    return frozenset(callers)


def _calls(agent: Agent, private_state, kind: str) -> bool:
    should_call = getattr(agent, "should_call", None)
    return bool(should_call(private_state, kind)) if should_call else False


def _inject_callers(state: GameState, *, grand: frozenset[int]) -> GameState:
    return GameState(
        hands=state.hands,
        public=replace(state.public, grand_tichu_callers=grand),
    )
