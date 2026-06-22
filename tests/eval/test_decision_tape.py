"""Decision-tape recorder: per Play Decision, ranked alternatives + choice."""

from __future__ import annotations

import dataclasses

from tichu_engine.legality import DragonGive, MahjongWish, SchupfenPass, legal_actions_for
from tichu_engine.state import deal_initial_state
from tichu_eval.decision_tape import (
    TapeSink, record_round, render_dragon, render_public_context, render_schupfen,
    render_text, render_wish,
)
from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_ml.rule_agent import RuleAgent


class _ScoringRule(RuleAgent):
    """RuleAgent (real, legal play) that also exposes uniform play_action_scores,
    so the tape has ranked alternatives to record."""

    def play_action_scores(self, private_state):
        legal = list(legal_actions_for(private_state))
        if not legal:
            return []
        p = 1.0 / len(legal)
        return [(a, p) for a in legal]


def test_tape_records_decisions_with_alternatives():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    agents = tuple(_ScoringRule() for _ in range(4))
    sink = TapeSink()
    record_round(agents, pos, round_idx=0, sink=sink, top_k=5)

    assert sink.records, "expected at least one reviewable decision"
    for rec in sink.records:
        assert len(rec.topk) >= 2            # only multi-option decisions recorded
        assert len(rec.topk) <= 5            # top_k cap
        assert rec.hand                      # acting seat's hand captured
        assert rec.chosen                    # a chosen action string
        assert 0 <= rec.seat < 4


def test_render_record_includes_public_context():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    agents = tuple(_ScoringRule() for _ in range(4))
    sink = TapeSink()
    record_round(agents, pos, round_idx=0, sink=sink, top_k=5)
    text = render_text(sink)
    assert "public:" in text and "trick:" in text


def test_render_text_is_reviewable():
    pos = generate_full_position_pool(seed=1, n=1)[0]
    agents = tuple(_ScoringRule() for _ in range(4))
    sink = TapeSink()
    record_round(agents, pos, round_idx=0, sink=sink)
    text = render_text(sink)
    assert "Round 0" in text
    assert "chose:" in text and "hand:" in text


class _AuxStub:
    """Stub exposing the diagnostic score methods for the non-Play Decisions."""
    def __init__(self, wish=None, dragon=None, schupfen=None):
        self._wish, self._dragon, self._schupfen = wish, dragon, schupfen

    def wish_action_scores(self, ps):
        return self._wish or []

    def dragon_action_scores(self, ps):
        return self._dragon or []

    def schupfen_action_scores(self, ps):
        return self._schupfen or {}


def _ps():
    st = deal_initial_state(seed=0)
    return st.private_view(st.public.current_player)


def test_render_wish_shows_choice_alternatives_and_call_context():
    ps = _ps()
    ps = dataclasses.replace(ps, public=dataclasses.replace(
        ps.public, grand_tichu_callers=frozenset({1})))
    agent = _AuxStub(wish=[(MahjongWish(14), 0.6), (MahjongWish(5), 0.4)])
    out = render_wish(agent, ps, MahjongWish(5), header="H")
    assert "WISH" in out and "chose=5" in out
    assert "14:0.60" in out                 # ranked alternative shown
    assert "grand_callers=[1]" in out       # the signal a Wish should react to


def test_render_wish_includes_public_context():
    ps = _ps()
    agent = _AuxStub(wish=[(MahjongWish(14), 0.6), (MahjongWish(5), 0.4)])
    out = render_wish(agent, ps, MahjongWish(14), header="H")
    assert "public:" in out and "trick:" in out


def test_render_dragon_shows_target():
    agent = _AuxStub(dragon=[(DragonGive(1), 0.7), (DragonGive(3), 0.3)])
    out = render_dragon(agent, _ps(), DragonGive(1))
    assert "DRAGON" in out and "chose=seat1" in out and "seat3:0.30" in out


def test_render_dragon_includes_public_context():
    agent = _AuxStub(dragon=[(DragonGive(1), 0.7), (DragonGive(3), 0.3)])
    out = render_dragon(agent, _ps(), DragonGive(1))
    assert "public:" in out and "trick:" in out


def test_render_schupfen_shows_each_direction():
    ps = _ps()
    cards = list(ps.hand)[:3]
    scores = {d: [(cards[0], 0.8), (cards[1], 0.2)] for d in ("next", "partner", "previous")}
    agent = _AuxStub(schupfen=scores)
    action = SchupfenPass(to_next=cards[0], to_partner=cards[1], to_previous=cards[2])
    out = render_schupfen(agent, ps, action)
    assert "SCHUPFEN" in out
    for d in ("next", "partner", "previous"):
        assert d in out
    assert "cand:" in out


def test_public_context_marks_acting_seat_and_shows_call_signal():
    ps = _ps()
    ps = dataclasses.replace(ps, public=dataclasses.replace(
        ps.public, scores=(20, -10), tichu_callers=frozenset({2}),
        grand_tichu_callers=frozenset({1}), mahjong_wish=9))
    out = render_public_context(ps, seat=ps.player)
    assert "scores=20/-10" in out
    assert "tichu=[2]" in out and "grand=[1]" in out
    assert "wish=9" in out
    # The acting seat is marked in the absolute-seat hand_sizes tuple.
    assert f"*{ps.public.hand_sizes[ps.player]}" in out


def test_public_context_renders_trick_sequence_and_passes():
    from tichu_engine.cards import Card
    from tichu_engine.combinations import Single
    from tichu_engine.state import Play, Trick

    ps = _ps()
    norm = [c for c in ps.hand if isinstance(c, Card)]
    a, b = norm[0], norm[1]
    trick = Trick(
        plays=(Play(player=1, combination=Single(a)),
               Play(player=2, combination=Single(b))),
        leader=2, passes=frozenset({3}),
    )
    ps = dataclasses.replace(ps, public=dataclasses.replace(ps.public, trick=trick))
    out = render_public_context(ps, seat=ps.player)
    assert "trick:" in out
    assert "lead=seat2" in out          # current top holder
    assert "seat1" in out               # the opening play's seat
    assert "passed" in out and "3" in out   # seat 3 has passed this trick


def test_public_context_lead_when_trick_empty():
    out = render_public_context(_ps(), seat=0)
    assert "trick:" in out
    assert "(lead)" in out              # empty trick reads as a fresh lead


def test_render_schupfen_includes_public_context():
    ps = _ps()
    cards = list(ps.hand)[:3]
    scores = {d: [(cards[0], 0.8)] for d in ("next", "partner", "previous")}
    agent = _AuxStub(schupfen=scores)
    action = SchupfenPass(to_next=cards[0], to_partner=cards[1], to_previous=cards[2])
    out = render_schupfen(agent, ps, action)
    assert "public:" in out and "trick:" in out


def test_agent_without_scores_records_nothing():
    pos = generate_full_position_pool(seed=0, n=1)[0]
    agents = tuple(RuleAgent() for _ in range(4))   # no play_action_scores
    sink = TapeSink()
    record_round(agents, pos, round_idx=0, sink=sink)
    assert sink.records == []
