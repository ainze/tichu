"""MLAgent — exported TS model + version assertion + fallback."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from tichu_engine.legality import legal_actions_for
from tichu_engine.state import deal_initial_state

from tichu_export.torchscript import export_torchscript
from tichu_inference.ml_agent import MLAgent, VersionMismatchError as MLVersionMismatch


# Track the live module constants so the fixture stays current across
# featurizer / action-space version bumps. Hard-coding "v1" here used
# to silently work because MLAgent's expected-version pin also tracked
# the live constants — both halves moved together. Now we keep the
# fixture explicit about that contract.
from tichu_training.action_space import ACTION_SPACE_VERSION as _ACTION_SPACE_VERSION
from tichu_training.featurizer import FEATURIZER_VERSION as _FEATURIZER_VERSION


class _DummyPolicy(torch.nn.Module):
    """Bare-minimum BCModel-shaped module — uses real featurizer output dim."""

    def __init__(self, feature_dim: int, action_space_size: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(feature_dim, action_space_size)
        self.pass_card_head = torch.nn.Linear(feature_dim, 3)
        self.wish_head = torch.nn.Linear(feature_dim, 14)
        self.dragon_head = torch.nn.Linear(feature_dim, 2)

    def forward(self, features, skill_decile):
        return {
            "play": self.fc(features),
            "schupfen": self.pass_card_head(features),
            "wish": self.wish_head(features),
            "dragon_assignment": self.dragon_head(features),
        }


def _export_dummy(tmp_path: Path, *, feature_dim: int, action_space_size: int,
                  featurizer_version: str = _FEATURIZER_VERSION,
                  action_space_version: str = _ACTION_SPACE_VERSION) -> Path:
    torch.manual_seed(0)
    model = _DummyPolicy(feature_dim, action_space_size)
    out = tmp_path / "policy.pt"
    export_torchscript(
        model,
        example_inputs=(torch.randn(1, feature_dim), torch.tensor([0], dtype=torch.long)),
        featurizer_version=featurizer_version,
        action_space_version=action_space_version,
        output_path=out,
    )
    return out


def test_act_returns_legal_action_on_deal_initial(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path,
                             feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    action = agent.act(ps)
    assert action in legal_actions_for(ps)


def test_version_mismatch_raises_at_construction(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    bad = _export_dummy(tmp_path,
                        feature_dim=FEATURIZER_OUTPUT_DIM,
                        action_space_size=ACTION_SPACE_SIZE,
                        featurizer_version="v999")
    with pytest.raises(MLVersionMismatch):
        MLAgent(bad)


def test_fallback_fires_on_model_exception(tmp_path, caplog):
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    # Broken artifact — wrong output dim breaks the action-space lookup.
    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM, action_space_size=42)
    # Bypass version check to load it: stamp matching versions then deliberately
    # poke the agent's internal module to raise.
    agent = MLAgent(artifact)
    # Monkey-patch the loaded module to always raise.
    def boom(*_a, **_k):
        raise RuntimeError("synthetic failure")
    agent._module = type("Broken", (), {"__call__": boom})()
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    with caplog.at_level(logging.ERROR, logger="tichu_inference.ml_agent"):
        action = agent.act(ps)
    assert action in legal_actions_for(ps)
    assert agent.last_fallback_used is True
    assert any("synthetic failure" in r.getMessage() or "fallback" in r.getMessage().lower()
               for r in caplog.records)


def test_fallback_fires_on_nan_logits(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path,
                             feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)

    nan_size = ACTION_SPACE_SIZE
    class NanModel:
        def __call__(self, features, skill_decile):
            n = features.shape[0]
            return {
                "play": torch.full((n, nan_size), float("nan")),
                "schupfen": torch.zeros(n, 3),
                "wish": torch.zeros(n, 14),
                "dragon_assignment": torch.zeros(n, 2),
            }
    agent._module = NanModel()

    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    action = agent.act(ps)
    assert action in legal_actions_for(ps)
    assert agent.last_fallback_used is True


class _FixedCall:
    """Call module returning a constant P(call) via logits [0, logit(p)]."""
    def __init__(self, p_call: float) -> None:
        import math
        self._call_logit = math.log(p_call / (1.0 - p_call))

    def __call__(self, features, skill):
        n = features.shape[0]
        return torch.tensor([[0.0, self._call_logit]]).repeat(n, 1)


def _bomb_state(*, leader, hand_extra=frozenset(), tichu_callers=frozenset(),
                grand_tichu_callers=frozenset()):
    """Seat 0 follows a King single with a 4x5 bomb in hand (+ hand_extra).
    `leader` is who holds the trick top: 2 = partner, 1 = opponent."""
    from tichu_engine.cards import Card, Suit
    from tichu_engine.combinations import FourOfAKindBomb, Single
    from tichu_engine.legality import _cards_in
    from tichu_engine.state import Play, PrivateState, PublicState, Trick

    bomb = FourOfAKindBomb(Card(suit=Suit.JADE, rank=5), Card(suit=Suit.SWORD, rank=5),
                           Card(suit=Suit.PAGODA, rank=5), Card(suit=Suit.STAR, rank=5))
    top = Single(Card(suit=Suit.SWORD, rank=13))
    hand = frozenset(_cards_in(bomb)) | frozenset(hand_extra)
    trick = Trick(plays=(Play(player=leader, combination=top),), leader=leader)
    pub = PublicState(
        current_player=0, hand_sizes=(len(hand), 5, 5, 5), scores=(0, 0), trick=trick,
        tichu_callers=tichu_callers, grand_tichu_callers=grand_tichu_callers,
    )
    return PrivateState(player=0, hand=hand, public=pub), bomb


def test_guard_suppresses_bomb_on_partners_trick():
    from tichu_inference.ml_agent import suppress_partner_trick_bomb
    pv, bomb = _bomb_state(leader=2)
    assert suppress_partner_trick_bomb(pv, bomb, legal_actions_for(pv)) is True


def test_guard_allows_bombing_an_opponent_trick():
    from tichu_inference.ml_agent import suppress_partner_trick_bomb
    pv, bomb = _bomb_state(leader=1)
    assert suppress_partner_trick_bomb(pv, bomb, legal_actions_for(pv)) is False


def test_guard_allows_non_bomb_actions():
    from tichu_engine.cards import Card, Suit
    from tichu_engine.combinations import Single
    from tichu_inference.ml_agent import suppress_partner_trick_bomb
    ace = Card(suit=Suit.JADE, rank=14)
    pv, _ = _bomb_state(leader=2, hand_extra={ace})
    assert suppress_partner_trick_bomb(pv, Single(ace), legal_actions_for(pv)) is False


def test_guard_carveout_lets_a_caller_go_out_on_the_bomb():
    # Seat 0 called Tichu and the bomb is its whole hand: playing it goes out and
    # banks the call — the one justified bomb over the partner's top.
    from tichu_inference.ml_agent import suppress_partner_trick_bomb
    pv, bomb = _bomb_state(leader=2, tichu_callers=frozenset({0}))
    assert suppress_partner_trick_bomb(pv, bomb, legal_actions_for(pv)) is False


def test_guard_still_fires_for_a_caller_not_going_out():
    # Caller, but the bomb leaves a card behind -> no go-out, guard still applies.
    from tichu_engine.cards import Card, Suit
    from tichu_inference.ml_agent import suppress_partner_trick_bomb
    pv, bomb = _bomb_state(leader=2, tichu_callers=frozenset({0}),
                           hand_extra={Card(suit=Suit.JADE, rank=3)})
    assert suppress_partner_trick_bomb(pv, bomb, legal_actions_for(pv)) is True


def test_act_applies_partner_trick_guard(tmp_path, monkeypatch):
    # Wire-through: with the policy ranking the bomb first, act() must return Pass
    # when the partner holds the trick top — and the bomb when the guard is off.
    import tichu_inference.ml_agent as ml
    from tichu_engine.legality import Pass
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM

    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    pv, bomb = _bomb_state(leader=2)

    def bomb_first(legal, logits):
        return sorted(legal, key=lambda a: 0 if a == bomb else 1)
    monkeypatch.setattr(ml, "_rank_legal_by_logits", bomb_first)

    assert isinstance(MLAgent(artifact).act(pv), Pass)
    assert MLAgent(artifact, partner_trick_guard=False).act(pv) == bomb


def test_call_threshold_controls_should_call(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)

    # P(call) = 0.30: call when threshold <= 0.30, decline above.
    a_low = MLAgent(artifact, tichu_threshold=0.25)
    a_low._tichu_call = _FixedCall(0.30)
    assert a_low.should_call(ps, "tichu") is True

    a_hi = MLAgent(artifact, tichu_threshold=0.35)
    a_hi._tichu_call = _FixedCall(0.30)
    assert a_hi.should_call(ps, "tichu") is False

    # Default threshold 0.5 reproduces argmax: P(call)=0.30 -> decline.
    a_def = MLAgent(artifact)
    a_def._tichu_call = _FixedCall(0.30)
    assert a_def.should_call(ps, "tichu") is False


def test_call_threshold_out_of_range_raises(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    with pytest.raises(ValueError):
        MLAgent(artifact, tichu_threshold=0.0)
    with pytest.raises(ValueError):
        MLAgent(artifact, grand_threshold=1.0)


def test_wish_action_scores_softmax_on_mahjong_state(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    from tichu_engine.cards import MAHJONG
    from tichu_engine.combinations import Single
    from tichu_engine.engine import step
    from tichu_engine.state import MahjongWishPending

    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)
    state = deal_initial_state(seed=0)
    # The Mahjong holder leads; playing it triggers the Wish pending decision.
    state2, _, _, _ = step(state, Single(MAHJONG))
    ps = state2.private_view(state2.public.current_player)
    assert isinstance(ps.public.pending_decision, MahjongWishPending)

    scores = agent.wish_action_scores(ps)
    assert scores, "expected legal wish ranks"
    assert all(0.0 <= p <= 1.0 for _, p in scores)
    assert abs(sum(p for _, p in scores) - 1.0) < 1e-5      # a softmax
    assert scores == sorted(scores, key=lambda t: -t[1])    # sorted desc


def test_rank_actions_returns_legal_actions(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path,
                             feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    agent = MLAgent(artifact)
    state = deal_initial_state(seed=0)
    ps = state.private_view(state.public.current_player)
    ranked = agent.rank_actions(ps)
    assert ranked is not None and len(ranked) > 0
    legal = legal_actions_for(ps)
    for a in ranked:
        assert a in legal


# --- team_scores round-local normalisation (inference-side fix) --------------
# The nets train on round-local scores (every round starts at 0; trick points
# accrue to both `scores` and `round_points_by_player`), but /act carries the
# GAME-cumulative score. MLAgent must feed the round-local team score —
# (rp[0]+rp[2], rp[1]+rp[3]) — so served decisions match the training regime.

def _play_state_with(scores, round_points):
    """A real post-schupfen play state with the given team scores + per-player
    round points (only those two public fields vary)."""
    import dataclasses
    state = deal_initial_state(seed=0)
    pub = dataclasses.replace(
        state.public, scores=scores, round_points_by_player=round_points,
    )
    return dataclasses.replace(state, public=pub).private_view(pub.current_player)


def _build_dummy_agent(tmp_path):
    from tichu_training.action_space import ACTION_SPACE_SIZE
    from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM
    artifact = _export_dummy(tmp_path, feature_dim=FEATURIZER_OUTPUT_DIM,
                             action_space_size=ACTION_SPACE_SIZE)
    return MLAgent(artifact)


def test_decisions_invariant_to_cumulative_score_offset(tmp_path):
    agent = _build_dummy_agent(tmp_path)
    rp = (10, 5, 20, 5)  # team0 round-local = 30, team1 = 10
    round_local = _play_state_with((30, 10), rp)
    cumulative = _play_state_with((630, 410), rp)   # same round, +600/+400 banked earlier
    assert agent.play_action_scores(round_local) == agent.play_action_scores(cumulative)


def test_agent_feeds_round_local_team_scores(tmp_path):
    # The team_scores section the model actually receives must be the round-local
    # value (team-summed round_points / 1000) — NOT the cumulative /act scores and
    # NOT a crude zero. Reading the fed feature slice isolates team_scores (#3)
    # from the separate round_points section (#4).
    from tichu_training.featurizer import SECTION_OFFSETS
    agent = _build_dummy_agent(tmp_path)
    ps = _play_state_with((900, 50), (10, 5, 20, 5))  # round-local (30, 10)
    feats, _ = agent._inputs(ps)
    off = SECTION_OFFSETS["team_scores"]
    assert feats[0, off:off + 2].tolist() == pytest.approx([0.030, 0.010])


def test_round_local_team_scores_pairs_teams_0_2_and_1_3():
    # Pure arithmetic + team pairing (seats {0,2} vs {1,3}); independent of scores.
    from tichu_inference.ml_agent import round_local_team_scores
    from tichu_engine.state import deal_initial_state
    import dataclasses
    pub = dataclasses.replace(
        deal_initial_state(seed=0).public,
        scores=(777, 0),  # cumulative — must be ignored
        round_points_by_player=(10, 5, 20, 7),
    )
    assert round_local_team_scores(pub) == (30, 12)
