r"""PROTOTYPE / SPIKE — THROWAWAY. PIMC search throughput feasibility (ADR-0030).

QUESTION it answers
  Does the on-turn, single-rooted PIMC loop — determinize -> PUCT/MCTS ->
  Leaf Rollout -> root-parallel aggregate — run end-to-end on the *real* engine,
  and what is the non-batchable per-game Python throughput? ADR-0029/0030 flag
  that residue (featurize/legal-mask/Resolver/engine `step`) as the unknown to
  profile before the real build. Output: sims/sec, engine-steps/sec, and a rough
  positions/hour extrapolation that sets the v1 sim budget.

What it does NOT answer — PLAY STRENGTH.
  The policy here is a RANDOM-LEGAL STUB: no torch, no `master` checkpoint. So the
  PUCT prior is uniform and the Leaf Rollout is a random playout. Move quality is
  meaningless — only loop correctness and timing are real. Swapping in the batched
  `master` policy (prior + rollout) is the real build, not this spike.

THROWAWAY SHORTCUTS (do not copy into the real build)
  * Works in ConcreteAction space — no Intent/Resolver layer (fine for timing).
  * Belief-OFF determinization (uniform constraint-respecting sampler).
  * Calls/Schupfen skipped (`deal_initial_state` start); round_outcome = card-point
    team delta only, normalized by 100. We are timing, not scoring.
  * On-turn only — no BombInterrupt solicitation (matches v1 scope).

Usage (PowerShell):
  $env:PYTHONPATH = (Resolve-Path src).Path
  py -3.14 scripts\spike_pimc_throughput.py --worlds 10 --sims 50 --seed 0
  py -3.14 scripts\spike_pimc_throughput.py --worlds 10 --sims 50 --full-round
"""

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import time
from collections import defaultdict

# Bootstrap: make the spike runnable with no env fiddling (in-harness PYTHONPATH
# can resolve to a sibling worktree — see auto-memory). Prepend THIS worktree's src.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tichu_engine.combinations import FourOfAKindBomb, StraightFlushBomb
from tichu_engine.engine import step
from tichu_engine.legality import legal_actions
from tichu_engine.state import GameState, PrivateState, deal_initial_state
from tichu_training.search.determinize import sample_determinized_world


class Counter:
    """Mutable engine-step tally, threaded through the search for throughput."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps = 0


def _is_bomb(action) -> bool:
    return isinstance(action, (FourOfAKindBomb, StraightFlushBomb))


def _round_outcome(state: GameState, root: int) -> float:
    """Team-relative round result for the root seat, normalized to ~[-1, 1]."""
    t = root % 2
    s = state.public.scores
    return (s[t] - s[1 - t]) / 100.0


def _advance_to_root_decision(state, root, rng, counter):
    """Step the env (other seats + any pending decision, all random-stub) until it
    is the root's Play decision (current_player == root, no pending) or the round
    ends. Returns (state, done)."""
    while not (state.public.current_player == root
               and state.public.pending_decision is None):
        act = rng.choice(tuple(legal_actions(state)))
        state, _, done, _ = step(state, act)
        counter.steps += 1
        if done:
            return state, True
    return state, False


def _leaf_rollout(state, root, rng, counter) -> float:
    """Random-policy playout to round-terminal; return normalized round_outcome."""
    while True:
        act = rng.choice(tuple(legal_actions(state)))
        state, _, done, _ = step(state, act)
        counter.steps += 1
        if done:
            return _round_outcome(state, root)


class Node:
    """A root Play-decision state. Edges = the root's legal ConcreteActions."""

    __slots__ = ("state", "actions", "N", "W", "nsum", "children")

    def __init__(self, state: GameState) -> None:
        self.state = state
        self.actions = tuple(legal_actions(state))
        self.N: dict = defaultdict(int)
        self.W: dict = defaultdict(float)
        self.nsum = 0
        self.children: dict = {}  # action -> Node | ("TERMINAL", value)


def _step_through(state, action, root, rng, counter):
    """Apply a root action, then advance the env to the next root decision.
    Returns (next_state, done, terminal_value_or_None)."""
    state, _, done, _ = step(state, action)
    counter.steps += 1
    if done:
        return state, True, _round_outcome(state, root)
    state, done = _advance_to_root_decision(state, root, rng, counter)
    if done:
        return state, True, _round_outcome(state, root)
    return state, False, None


def _select(node: Node, c_puct: float, rng) -> object:
    unvisited = [a for a in node.actions if node.N[a] == 0]
    if unvisited:
        return rng.choice(unvisited)
    sqrt_total = math.sqrt(node.nsum)
    prior = 1.0 / len(node.actions)  # uniform — stub policy
    return max(
        node.actions,
        key=lambda a: node.W[a] / node.N[a]
        + c_puct * prior * sqrt_total / (1 + node.N[a]),
    )


def _simulate(node: Node, root, rng, counter, c_puct) -> float:
    a = _select(node, c_puct, rng)
    child = node.children.get(a)
    if child is None:
        next_state, done, term_val = _step_through(node.state, a, root, rng, counter)
        if done:
            node.children[a] = ("TERMINAL", term_val)
            value = term_val
        else:
            child_node = Node(next_state)
            node.children[a] = child_node
            value = _leaf_rollout(next_state, root, rng, counter)  # expand: one rollout
    elif isinstance(child, tuple):  # ("TERMINAL", value)
        value = child[1]
    else:
        value = _simulate(child, root, rng, counter, c_puct)
    node.N[a] += 1
    node.W[a] += value
    node.nsum += 1
    return value


def pimc_search(own_view: PrivateState, worlds, sims, rng, counter, c_puct=1.4):
    """Root-parallel PIMC: K determinized worlds, N sims each, sum root visits."""
    root = own_view.player
    agg_visits: dict = defaultdict(int)
    agg_value: dict = defaultdict(float)
    for _ in range(worlds):
        world = sample_determinized_world(own_view, None, rng)  # belief-off
        node = Node(world)
        for _ in range(sims):
            _simulate(node, root, rng, counter, c_puct)
        for a in node.actions:
            agg_visits[a] += node.N[a]
            agg_value[a] += node.W[a]
    best = max(agg_visits, key=lambda a: agg_visits[a])
    return best, agg_visits, agg_value


def _fmt_action(a) -> str:
    name = type(a).__name__
    bomb = " [BOMB]" if _is_bomb(a) else ""
    return f"{name}{bomb}"


def _one_decision(args) -> None:
    rng = random.Random(args.seed)
    counter = Counter()
    state = deal_initial_state(seed=args.seed)
    root = state.public.current_player
    own_view = state.private_view(root)

    n_bombs = sum(1 for a in legal_actions(state) if _is_bomb(a))
    print(f"Opening decision - root seat {root}, "
          f"{len(own_view.hand)} cards, {len(tuple(legal_actions(state)))} legal "
          f"actions ({n_bombs} bombs).")

    t0 = time.perf_counter()
    best, visits, value = pimc_search(own_view, args.worlds, args.sims, rng, counter)
    wall = time.perf_counter() - t0

    total_sims = args.worlds * args.sims
    print(f"\nsearched {args.worlds} worlds x {args.sims} sims = {total_sims} sims "
          f"in {wall:.3f}s")
    print(f"  sims/sec        : {total_sims / wall:,.0f}")
    print(f"  engine steps    : {counter.steps:,}  ({counter.steps / wall:,.0f}/sec)")
    print(f"  one decision    : {wall * 1000:.1f} ms")

    print("\ntop root actions by aggregated visits:")
    for a in sorted(visits, key=visits.get, reverse=True)[:5]:
        q = value[a] / visits[a] if visits[a] else 0.0
        mark = "  <- chosen" if a == best else ""
        print(f"  {_fmt_action(a):20s} visits={visits[a]:5d}  Q={q:+.3f}{mark}")

    _extrapolate(wall, args)


def _full_round(args) -> None:
    """Time a whole round where the root seat SEARCHES every Play decision and the
    other three seats play random-stub — a realistic per-round throughput read."""
    rng = random.Random(args.seed)
    counter = Counter()
    state = deal_initial_state(seed=args.seed)
    root = state.public.current_player

    decisions = 0
    t0 = time.perf_counter()
    while True:
        if state.public.current_player == root and state.public.pending_decision is None:
            own_view = state.private_view(root)
            best, _, _ = pimc_search(own_view, args.worlds, args.sims, rng, counter)
            state, _, done, _ = step(state, best)
            counter.steps += 1
            decisions += 1
        else:
            act = rng.choice(tuple(legal_actions(state)))
            state, _, done, _ = step(state, act)
            counter.steps += 1
        if done:
            break
    wall = time.perf_counter() - t0

    print(f"Full round - root seat {root} searched {decisions} Play decisions "
          f"(others random) in {wall:.3f}s")
    print(f"  per searched decision (avg): {wall / decisions * 1000:.1f} ms")
    print(f"  engine steps   : {counter.steps:,}  ({counter.steps / wall:,.0f}/sec)")
    _extrapolate(wall / decisions, args, per_round=wall, decisions=decisions)


def _extrapolate(per_decision_s, args, per_round=None, decisions=None) -> None:
    print("\nrough extrapolation (STUB policy — real master forward adds batched GPU cost):")
    dph = 3600.0 / per_decision_s
    print(f"  ~{dph:,.0f} search-decisions/hour at {args.worlds}x{args.sims} sims")
    if per_round is not None:
        # A seat-swapped Tournament position ~= 2 rounds; the search TEAM has 2
        # searching seats, so ~2x the single-seat decision load per round. Rough.
        position_s = per_round * 2 * 2
        print(f"  ~{3600.0 / position_s:,.1f} Tournament positions/hour "
              f"(assumes 2 rounds/position x 2 searching seats; ~{decisions} dec/round/seat)")
        print("  -> a 200-position pool ~= "
              f"{200 * position_s / 3600.0:,.1f} h single-process "
              "(divide by worker count; ADR-0026 process pool).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worlds", type=int, default=10, help="K determinized worlds")
    ap.add_argument("--sims", type=int, default=50, help="MCTS sims per world")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-round", action="store_true",
                    help="time a whole round (root searches every Play decision)")
    args = ap.parse_args()

    print("=== PIMC throughput spike (THROWAWAY, stub policy) ===")
    if args.full_round:
        _full_round(args)
    else:
        _one_decision(args)


if __name__ == "__main__":
    main()
