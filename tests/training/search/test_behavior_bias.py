"""Behavior Bias — δ on the play logits of one lever's target actions, in that
lever's situation only. Validated through the real legality path on hand-built
states; logits are synthetic (no torch, no checkpoints). See `behavior_bias`."""

import numpy as np
import pytest

from tichu_engine.cards import DOG, PHOENIX, Card, Suit
from tichu_engine.combinations import Pair, Single
from tichu_engine.legality import Pass, legal_actions_for
from tichu_engine.state import Play, PrivateState, PublicState, Trick
from tichu_training.featurizer import _combination_to_action_index
from tichu_training.search.behavior_bias import LEVERS, bias_play_logits, lever_gap

N = 1809


def _c(rank, suit=Suit.JADE):
    return Card(suit=suit, rank=rank)


def _pv(hand, *, leader=None, top=None, callers=frozenset(), player=0):
    sizes = [5, 5, 5, 5]
    sizes[player] = len(hand)
    trick = Trick.empty() if leader is None else Trick(
        plays=(Play(player=leader, combination=top),), leader=leader)
    pub = PublicState(current_player=player, hand_sizes=tuple(sizes), scores=(0, 0),
                      trick=trick, tichu_callers=frozenset(callers))
    return PrivateState(player=player, hand=frozenset(hand), public=pub)


def _logits(scores: dict):
    """Logits that put `scores[action]` on each action's index, -5 elsewhere."""
    out = np.full(N, -5.0, dtype=np.float32)
    for action, s in scores.items():
        out[_combination_to_action_index(action)] = s
    return out


def _argmax(pv, logits):
    legal = list(legal_actions_for(pv))
    return max(legal, key=lambda a: logits[_combination_to_action_index(a)])


# --- pass_vs_opponent ---------------------------------------------------------

FOLLOW_OPP = dict(leader=1, top=Single(_c(5, Suit.SWORD)))


def test_pass_bias_moves_only_the_pass_logit_when_an_opponent_holds_the_trick():
    pv = _pv([_c(9), _c(12)], **FOLLOW_OPP)
    base = _logits({Pass(): 1.0, Single(_c(9)): 2.0, Single(_c(12)): 0.5})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["pass_vs_opponent"], 1.5)
    diff = out - base
    assert diff[_combination_to_action_index(Pass())] == pytest.approx(1.5)
    assert np.count_nonzero(diff) == 1
    assert isinstance(_argmax(pv, out), Pass)          # 2.5 now beats 2.0


def test_pass_bias_does_not_fire_when_the_partner_holds_the_trick():
    pv = _pv([_c(9), _c(12)], leader=2, top=Single(_c(5, Suit.SWORD)))
    base = _logits({Pass(): 1.0, Single(_c(9)): 2.0})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["pass_vs_opponent"], 5.0)
    assert np.array_equal(out, base)


def test_bias_never_mutates_the_input_logits():
    pv = _pv([_c(9), _c(12)], **FOLLOW_OPP)
    base = _logits({Pass(): 1.0})
    before = base.copy()
    bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["pass_vs_opponent"], 3.0)
    assert np.array_equal(base, before)


def test_zero_delta_is_the_identity():
    pv = _pv([_c(9), _c(12)], **FOLLOW_OPP)
    base = _logits({Pass(): 1.0, Single(_c(9)): 2.0})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["pass_vs_opponent"], 0.0)
    assert np.array_equal(out, base)


# --- dog_lead -----------------------------------------------------------------

def test_dog_bias_fires_when_leading_with_the_dog_and_partner_not_called():
    pv = _pv([DOG, _c(9), _c(12)])
    base = _logits({Single(DOG): 3.0, Single(_c(9)): 2.0})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["dog_lead"], -1.5)
    assert out[_combination_to_action_index(Single(DOG))] == pytest.approx(1.5)
    assert _argmax(pv, out) == Single(_c(9))


def test_dog_bias_is_off_when_partner_has_called():
    pv = _pv([DOG, _c(9), _c(12)], callers={2})
    base = _logits({Single(DOG): 3.0})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["dog_lead"], -9.0)
    assert np.array_equal(out, base)


def test_dog_bias_is_off_when_following():
    pv = _pv([DOG, _c(9), _c(12)], **FOLLOW_OPP)
    base = _logits({Pass(): 1.0})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base, LEVERS["dog_lead"], -9.0)
    assert np.array_equal(out, base)


# --- single_lead --------------------------------------------------------------

def test_single_lead_bias_moves_every_non_dog_single_and_nothing_else():
    hand = [DOG, _c(9), _c(9, Suit.SWORD), _c(12)]
    pv = _pv(hand)
    legal = list(legal_actions_for(pv))
    base = np.zeros(N, dtype=np.float32)
    out = bias_play_logits(pv, legal, base, LEVERS["single_lead"], -2.0)
    for a in legal:
        expect = -2.0 if isinstance(a, Single) and a.card is not DOG else 0.0
        assert out[_combination_to_action_index(a)] == pytest.approx(expect), a


def test_a_shared_intent_index_is_biased_once_not_per_realisation():
    # Two natural 9s: two concrete Single(9) actions map to ONE Intent index.
    pv = _pv([_c(9), _c(9, Suit.SWORD), _c(12)])
    legal = list(legal_actions_for(pv))
    base = np.zeros(N, dtype=np.float32)
    out = bias_play_logits(pv, legal, base, LEVERS["single_lead"], 1.0)
    assert out.max() == pytest.approx(1.0)


# --- phoenix_in_combination ---------------------------------------------------

def test_phoenix_bias_targets_only_multi_card_combinations_with_the_phoenix():
    pv = _pv([PHOENIX, _c(9), _c(12)])
    legal = list(legal_actions_for(pv))
    base = np.zeros(N, dtype=np.float32)
    out = bias_play_logits(pv, legal, base, LEVERS["phoenix_in_combination"], 2.0)
    hit = [a for a in legal if out[_combination_to_action_index(a)] != 0]
    assert hit and all(isinstance(a, Pair) and PHOENIX in (a.a, a.b) for a in hit)
    assert all(out[_combination_to_action_index(a)] == 0
               for a in legal if isinstance(a, Single))


def test_phoenix_bias_is_off_with_no_phoenix_combination_legal():
    pv = _pv([PHOENIX, _c(9)], **FOLLOW_OPP)        # only singles over a single
    base = _logits({Pass(): 1.0})
    out = bias_play_logits(pv, list(legal_actions_for(pv)), base,
                           LEVERS["phoenix_in_combination"], 5.0)
    assert np.array_equal(out, base)


# --- forced / gap -------------------------------------------------------------

def test_a_forced_decision_is_never_biased():
    pv = _pv([_c(3)], **FOLLOW_OPP)                  # cannot beat: Pass only
    legal = list(legal_actions_for(pv))
    assert len(legal) == 1
    base = _logits({Pass(): 1.0})
    assert np.array_equal(bias_play_logits(pv, legal, base, LEVERS["pass_vs_opponent"], 5.0), base)


def test_gap_is_best_target_minus_best_other_and_none_off_situation():
    pv = _pv([_c(9), _c(12)], **FOLLOW_OPP)
    legal = list(legal_actions_for(pv))
    logits = _logits({Pass(): 1.0, Single(_c(9)): 2.5, Single(_c(12)): 0.5})
    assert lever_gap(pv, legal, logits, LEVERS["pass_vs_opponent"]) == pytest.approx(-1.5)
    assert lever_gap(pv, legal, logits, LEVERS["dog_lead"]) is None


def test_gap_predicts_exactly_which_delta_flips_the_argmax():
    pv = _pv([_c(9), _c(12)], **FOLLOW_OPP)
    legal = list(legal_actions_for(pv))
    logits = _logits({Pass(): 1.0, Single(_c(9)): 2.5, Single(_c(12)): 0.5})
    lever = LEVERS["pass_vs_opponent"]
    assert not isinstance(_argmax(pv, bias_play_logits(pv, legal, logits, lever, 1.4)), Pass)
    assert isinstance(_argmax(pv, bias_play_logits(pv, legal, logits, lever, 1.6)), Pass)


# --- δ calibration ------------------------------------------------------------

from tichu_training.search.behavior_bias import choose_delta  # noqa: E402


def test_choose_delta_flips_the_target_share_of_situations_in_each_direction():
    # 10 situations: 4 already pick the target (g > 0), 6 do not (g < 0).
    gaps = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, 0.3, 0.6, 1.0, 4.0]
    d = choose_delta(gaps, share=0.2, cap=8.0)
    # +δ must flip the 2 smallest negative margins (0.1, 0.2) -> just above 0.2.
    assert 0.2 < d["plus"] < 0.5 and d["plus_share"] == pytest.approx(0.2)
    # −δ must flip the 2 smallest positive margins (0.3, 0.6) -> just above 0.6.
    assert -1.0 < d["minus"] < -0.6 and d["minus_share"] == pytest.approx(0.2)


def test_choose_delta_caps_and_reports_the_share_it_can_reach():
    gaps = [-20.0, -15.0, 0.5, 0.7]          # only 2 of 4 below 0, both far
    d = choose_delta(gaps, share=0.25, cap=8.0)
    assert d["plus"] == 8.0 and d["plus_share"] == 0.0
    assert d["minus"] < -0.5 and d["minus_share"] == pytest.approx(0.25)


def test_choose_delta_share_is_what_the_flip_rule_predicts():
    rng = np.random.default_rng(0)
    gaps = rng.normal(0.5, 2.0, size=5000)
    d = choose_delta(gaps, share=0.1, cap=8.0)
    assert np.mean((gaps < 0) & (gaps > -d["plus"])) == pytest.approx(0.1, abs=1e-3)
    assert np.mean((gaps > 0) & (gaps < -d["minus"])) == pytest.approx(0.1, abs=1e-3)
