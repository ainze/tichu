"""Self-play-derived Belief-example emit (ADR-0041 pre-check).

The BSW path ([emit.py](../../../src/tichu_training/belief/emit.py)) reads labels
off a parsed replay; this path reads them off the live engine `GameState` during
champion self-play, so a Belief Model can be trained on the distribution it is
consumed on. Deterministic RuleAgents — no torch, no checkpoints.
"""

import random

import numpy as np

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_eval.play_full import play_full_round
from tichu_ml.rule_agent import RuleAgent
from tichu_training.belief.selfplay_emit import (
    belief_examples_for_selfplay_round,
    card_counting_marginals,
    belief_examples_for_selfplay_rounds,
    floor_marginals_for_example,
    load_belief_marginals_fn,
    load_examples,
    save_belief_checkpoint,
    save_examples,
    holder_nll,
    masked_bce,
    round_level_split,
    score_belief_top1,
    score_floor_top1,
    train_selfplay_belief,
    which_opponent_top1,
)
from tichu_training.card_slots import card_slot
from tichu_training.search.determinize import sample_determinized_world


def _agents():
    return [RuleAgent() for _ in range(4)]


def _position(seed=11):
    return generate_full_position_pool(seed=seed, n=1)[0]


def _decision_views(position):
    """The acting seat's `PrivateState` at every Play Decision."""
    seen = []
    play_full_round(
        _agents(), position.state, position.grand_prefixes,
        state_observer=lambda seat, game_state, _a: seen.append(
            game_state.private_view(seat)
        ),
    )
    return seen


def test_card_counting_floor_spreads_each_unseen_card_by_hand_size():
    """The floor knows only what is public: every **Unseen Card** is split across
    the three opponents in proportion to their `hand_sizes`, and cards the actor
    can see — own **Hand** or already played — carry no mass."""
    for view in _decision_views(_position()):
        probs = card_counting_marginals(view)

        rel_seats = ((view.player + 1) % 4, (view.player + 2) % 4, (view.player + 3) % 4)
        sizes = [view.public.hand_sizes[s] for s in rel_seats]
        total_unseen = sum(sizes)
        seen = {card_slot(c) for c in view.hand} | {
            card_slot(c) for c in view.public.played_cards_this_round
        }
        unseen = [s for s in range(56) if s not in seen]

        assert len(unseen) == total_unseen
        for slot in unseen:
            np.testing.assert_allclose(
                probs[:, slot], np.array(sizes, dtype=np.float32) / total_unseen,
                rtol=1e-6,
            )
        for slot in seen:
            np.testing.assert_array_equal(probs[:, slot], np.zeros(3, dtype=np.float32))


def _observed_decisions(position):
    """The (seat, true hands) of every Play Decision, observed independently of
    the emitter. RuleAgents are deterministic, so replaying the same Starting
    Position reproduces the same Decision sequence."""
    seen = []
    play_full_round(
        _agents(), position.state, position.grand_prefixes,
        state_observer=lambda seat, game_state, _action: seen.append(
            (seat, game_state.hands)
        ),
    )
    return seen


def test_selfplay_labels_place_every_unseen_card_with_its_true_holder():
    """One example per Play Decision; each Unseen Card is labelled against the
    opponent who actually holds it, in relative-seat order (next/partner/previous),
    and the mask covers exactly the Unseen Cards."""
    position = _position()
    expected = _observed_decisions(position)
    examples = belief_examples_for_selfplay_round(_agents(), position)

    assert expected, "fixture broken — no Play Decisions in the round"
    assert len(examples) == len(expected)
    for ex, (seat, hands) in zip(examples, expected):
        rel_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
        for opp_idx, opp_seat in enumerate(rel_seats):
            labelled = {s for s in range(56) if ex.labels[opp_idx, s]}
            assert labelled == {card_slot(c) for c in hands[opp_seat]}

        # The mask is the Unseen Cards: not in the actor's own Hand, not played.
        unseen = {card_slot(c) for s in rel_seats for c in hands[s]}
        own = {card_slot(c) for c in hands[seat]}
        assert {s for s in range(56) if ex.mask[0, s]} == unseen
        assert not (own & unseen)


def test_top1_scores_only_unseen_cards_and_is_perfect_on_ground_truth():
    """The discriminating metric: for each **Unseen Card**, did the predictor pick
    the opponent who actually holds it? Cards the actor can already see must not
    enter the denominator — otherwise a predictor is credited for knowing what
    everyone knows."""
    examples = belief_examples_for_selfplay_round(_agents(), _position())

    total_hits = total_cards = 0
    for ex in examples:
        unseen = int(ex.mask[0].sum())
        hits, cards = which_opponent_top1(ex.labels, ex)

        assert cards == unseen, "denominator must be the Unseen Cards only"
        assert hits == unseen, "ground truth as prediction must score every card"
        total_hits += hits
        total_cards += cards

    assert total_cards > 0, "fixture broken — no Unseen Cards scored"
    assert total_hits == total_cards


def test_floor_marginals_drive_a_legal_determinized_world():
    """The belief-off arm's contract: marginals feed `sample_determinized_world`
    directly. A **Determinized World** must keep the actor's **Hand** untouched
    and deal every **Unseen Card** out exactly once, respecting `hand_sizes` —
    the relative-seat convention is shared, not re-derived."""
    rng = random.Random(7)
    for view in _decision_views(_position()):
        world = sample_determinized_world(view, card_counting_marginals(view), rng)

        assert set(world.hands[view.player]) == set(view.hand)
        for seat in range(4):
            assert len(world.hands[seat]) == view.public.hand_sizes[seat]

        dealt = [c for s in range(4) for c in world.hands[s]]
        assert len(dealt) == len(set(dealt)), "a card was dealt to two seats"


def test_round_level_split_never_puts_one_round_on_both_sides():
    """Hidden hands are round-unique, so an example-level split leaks: the same
    deal would appear in train and holdout. The ADR-0033 post-mortem named this
    as what made round-unique features memorisation-prone. Whole Rounds move
    together, and every example carries the provenance that makes that possible."""
    positions = generate_full_position_pool(seed=3, n=6)
    examples = belief_examples_for_selfplay_rounds(_agents, positions)

    assert examples
    round_ids = {ex.round_id for ex in examples}
    assert len(round_ids) == len(positions), "each Round needs its own provenance"

    train, holdout = round_level_split(examples, holdout_frac=0.5, seed=0)

    assert train and holdout
    assert len(train) + len(holdout) == len(examples)
    assert not ({ex.round_id for ex in train} & {ex.round_id for ex in holdout})


def test_belief_trained_on_selfplay_beats_the_card_counting_floor():
    """The ADR-0041 pre-check, as a behaviour: a Belief Model fit on self-play
    identifies *which* opponent holds an Unseen Card better than a predictor that
    knows only `hand_sizes` — measured on **held-out Rounds** the fit never saw.
    If this fails, `D_on - D_off` is zero by construction and the gate is over."""
    positions = generate_full_position_pool(seed=17, n=250)
    examples = belief_examples_for_selfplay_rounds(_agents, positions)
    train, holdout = round_level_split(examples, holdout_frac=0.25, seed=0)

    model = train_selfplay_belief(train, epochs=8, seed=0)

    floor = score_floor_top1(holdout)
    trained = score_belief_top1(model, holdout)
    assert trained > floor + 0.03, (
        f"belief top1 {trained:.4f} vs floor {floor:.4f} — no supervision edge"
    )


def test_the_two_floors_agree():
    """The floor is computed two ways — from a live `PrivateState` (what the
    **Belief-Optimal Chooser** consumes) and from a `BeliefExample` (what scoring
    consumes). They must be the same predictor: a relative-seat mismatch between
    them would silently bias `D_off` while every other test still passed."""
    position = _position()
    views = _decision_views(position)
    examples = belief_examples_for_selfplay_round(_agents(), position)

    assert len(views) == len(examples)
    for view, ex in zip(views, examples):
        live = card_counting_marginals(view)
        hits_live, cards_live = which_opponent_top1(live, ex)
        assert (hits_live, cards_live) == which_opponent_top1(
            floor_marginals_for_example(ex), ex
        )
        np.testing.assert_allclose(live, floor_marginals_for_example(ex), atol=1e-6)


def floor_marginals_for_example(ex):
    card_mask = np.asarray(ex.mask)[0].astype(bool)
    unseen = int(card_mask.sum())
    sizes = np.asarray(ex.labels)[:, card_mask].sum(axis=1) / unseen
    probs = np.zeros((3, 56), dtype=np.float32)
    probs[:, card_mask] = sizes[:, None]
    return probs


def test_masked_bce_scores_only_unseen_cards():
    """Calibration, not argmax: the world sampler consumes full marginals, so a
    predictor can differ from the floor without winning more top-1s. Positions
    the actor can already see must not enter the average — including them would
    reward confidence about common knowledge."""
    examples = belief_examples_for_selfplay_round(_agents(), _position())

    for ex in examples:
        unseen = int(ex.mask[0].sum())
        certain = np.where(np.asarray(ex.labels) > 0.5, 1.0, 0.0).astype(np.float32)

        total, n = masked_bce(certain, ex)
        assert n == 3 * unseen, "every (opponent, Unseen Card) position is scored"
        assert total / max(1, n) < 1e-3, "ground truth must score ~0 loss"

        # A predictor that is confidently wrong everywhere must score far worse
        # than the floor, and only because of the masked-in positions.
        worse, _ = masked_bce(1.0 - certain, ex)
        assert worse > total


def test_training_early_stops_on_the_holdout_rather_than_running_to_the_last_epoch():
    """Hidden hands are round-unique, so a belief fit memorises deals if left to
    run: the ADR-0033 post-mortem's failure mode, reproduced here at train top-1
    0.998 / holdout below the floor. The fit must therefore be selected by
    held-out score per epoch, not by epoch count."""
    positions = generate_full_position_pool(seed=17, n=120)
    examples = belief_examples_for_selfplay_rounds(_agents, positions)
    train, holdout = round_level_split(examples, holdout_frac=0.3, seed=0)

    model, history = train_selfplay_belief(
        train, holdout=holdout, hidden=256, epochs=6, seed=0,
    )

    assert len(history) == 6, "one held-out reading per epoch"
    best = max(history, key=lambda row: row["holdout_top1"])
    assert score_belief_top1(model, holdout) == best["holdout_top1"], (
        "the returned model must be the best-holdout epoch, not the last"
    )


def test_examples_round_trip_through_disk(tmp_path):
    """Emitting is the expensive step, so examples are cached and re-read for
    capacity sweeps. What comes back must be what went in — labels, mask,
    features and Round provenance — or a sweep silently studies different data."""
    original = belief_examples_for_selfplay_round(_agents(), _position())
    path = tmp_path / "examples.npz"

    save_examples(path, original)
    restored = load_examples(path)

    assert len(restored) == len(original)
    for got, want in zip(restored, original):
        np.testing.assert_array_equal(got.features, want.features)
        np.testing.assert_array_equal(got.labels, want.labels)
        np.testing.assert_array_equal(got.mask, want.mask)
        assert got.cards_played == want.cards_played
        assert got.round_id == want.round_id


def test_a_saved_belief_checkpoint_becomes_chooser_ready_marginals(tmp_path):
    """The belief-on arm's wiring: a fitted checkpoint must come back as a
    `(3, 56)` probability grid the Determinization Sampler accepts, in the same
    relative-seat order as the floor — otherwise belief-on and belief-off would
    sample from differently-ordered worlds and `D_on - D_off` would be garbage."""
    positions = generate_full_position_pool(seed=17, n=40)
    examples = belief_examples_for_selfplay_rounds(_agents, positions)
    model = train_selfplay_belief(examples, hidden=64, epochs=1, seed=0)

    path = tmp_path / "belief.bin"
    save_belief_checkpoint(path, model)
    marginals_fn = load_belief_marginals_fn(path)

    rng = random.Random(3)
    for view in _decision_views(_position()):
        probs = marginals_fn(view)
        assert probs.shape == (3, 56)
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        world = sample_determinized_world(view, probs, rng)
        for seat in range(4):
            assert len(world.hands[seat]) == view.public.hand_sizes[seat]


def test_holder_nll_compares_the_distribution_the_sampler_actually_draws_from():
    """`_draw_holder` normalises its weights per card, so what governs sampling is
    the *relative* mass across the three opponents — not the absolute values BCE
    scores. Holder NLL measures exactly that, and is comparable between the
    independent-sigmoid and per-card-softmax parameterisations."""
    examples = belief_examples_for_selfplay_round(_agents(), _position())

    for ex in examples:
        unseen = int(ex.mask[0].sum())
        certain = np.where(np.asarray(ex.labels) > 0.5, 1.0, 0.0).astype(np.float32)

        total, n = holder_nll(certain, ex)
        assert n == unseen, "one 3-way term per Unseen Card"
        assert total / max(1, n) < 1e-3, "naming the true holder costs ~0"

        # The floor is uninformative about *which* opponent, so it must cost
        # about ln(3) per card when the three hands are the same size.
        floor_total, floor_n = holder_nll(floor_marginals_for_example(ex), ex)
        assert floor_total / max(1, floor_n) <= np.log(3) + 1e-6
