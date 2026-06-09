"""The torch-free divergence classifiers must bucket play decisions correctly — this is the
part most likely to have logic bugs. The agent/corpus glue in `main` is exercised by running
the script against real checkpoints (not unit-tested). Import works without torch because the
module's heavy imports are lazy inside `main`.
"""

from tichu_engine.cards import Card, DRAGON, Suit
from tichu_engine.combinations import FourOfAKindBomb, Pair, Single
from tichu_engine.legality import Pass
from tichu_engine.state import Play, PrivateState, PublicState, Trick

from scripts.mine_divergences import (
    action_kind,
    caller_ctx,
    cluster_key,
    disagreement_kind,
    following,
    handsize_bucket,
    is_premium,
    points_bucket,
)


def _c(suit, rank):
    return Card(suit=suit, rank=rank)


A = _c(Suit.JADE, 14)
K = _c(Suit.SWORD, 13)
FIVE = _c(Suit.JADE, 5)
KING_TEN = _c(Suit.STAR, 13)
BOMB = FourOfAKindBomb(_c(Suit.JADE, 7), _c(Suit.SWORD, 7), _c(Suit.PAGODA, 7), _c(Suit.STAR, 7))


def _pv(player, hand, *, leader=None, top=None, callers=frozenset()):
    if top is not None:
        trick = Trick(plays=(Play(player=leader, combination=top),), leader=leader)
    else:
        trick = Trick.empty()
    hand = frozenset(hand)
    sizes = [5, 5, 5, 5]
    sizes[player] = len(hand)
    pub = PublicState(current_player=player, hand_sizes=tuple(sizes), scores=(0, 0),
                      trick=trick, tichu_callers=callers)
    return PrivateState(player=player, hand=hand, public=pub)


def test_action_kind():
    assert action_kind(Single(A)) == "SINGLE"
    assert action_kind(Pair(A, _c(Suit.SWORD, 14))) == "PAIR"
    assert action_kind(Pass()) == "PASS"
    assert action_kind(BOMB) == "BOMB4"


def test_is_premium():
    assert is_premium(Single(DRAGON)) is True
    assert is_premium(Single(A)) is False
    assert is_premium(Pass()) is False


def test_disagreement_kind_pass_axes():
    assert disagreement_kind(Pass(), Single(K)) == "cotrain_contests"
    assert disagreement_kind(Single(K), Pass()) == "cotrain_cedes"


def test_disagreement_kind_bomb_and_premium():
    assert disagreement_kind(Single(A), BOMB) == "cotrain_bombs"
    assert disagreement_kind(Single(K), Single(DRAGON)) == "cotrain_premium"
    assert disagreement_kind(Single(DRAGON), Single(K)) == "human_premium"


def test_disagreement_kind_height_and_shape():
    assert disagreement_kind(Single(A), Single(K)) == "cotrain_lower"   # 13 < 14
    assert disagreement_kind(Single(K), Single(A)) == "cotrain_higher"  # 14 > 13
    assert disagreement_kind(Single(A), Pair(A, _c(Suit.SWORD, 14))) == "diff_shape"


def test_caller_ctx():
    assert caller_ctx(_pv(0, {A}, callers=frozenset({0}))) == "self_caller"
    assert caller_ctx(_pv(0, {A}, callers=frozenset({2}))) == "partner_caller"
    assert caller_ctx(_pv(0, {A}, callers=frozenset({1}))) == "opp_caller"
    assert caller_ctx(_pv(0, {A})) == "no_caller"


def test_following_and_buckets():
    lead = _pv(0, {A})
    assert following(lead) is False
    follow = _pv(0, {A}, leader=1, top=Single(K))
    assert following(follow) is True

    assert points_bucket(_pv(0, {A})) == "0"
    assert points_bucket(_pv(0, {A}, leader=1, top=Single(FIVE))) == "1-9"
    # A trick worth 5 (FIVE) + 10 (KING_TEN) = 15 points.
    two = Trick(plays=(Play(1, Single(FIVE)), Play(2, Single(KING_TEN))), leader=2)
    pub = PublicState(current_player=0, hand_sizes=(1, 4, 4, 4), scores=(0, 0), trick=two)
    assert points_bucket(PrivateState(player=0, hand=frozenset({A}), public=pub)) == "10+"

    assert handsize_bucket(_pv(0, {A})) == "1-3"
    assert handsize_bucket(_pv(0, {A, K, FIVE, _c(Suit.PAGODA, 9), _c(Suit.STAR, 3)})) == "4-7"


def test_cluster_key():
    pv = _pv(0, {A}, leader=1, top=Single(K), callers=frozenset({1}))
    assert cluster_key(pv, Pass(), Single(A)) == ("follow", "cotrain_contests", "opp_caller")
