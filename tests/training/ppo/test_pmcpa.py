"""pMCPA (ADR-0036): per-Round run-time policy adaptation on the paired-advantage
estimator. The collection roots MID-ROUND at a single actor's view, samples
Determinized Worlds from it, and emits luck-cancelled play-head rows for ONLY
that actor; the agent adapts a throwaway play net per Round and restores theta_o
exactly. Tiny nets, no checkpoints (mirrors test_vine.py)."""

import copy

import torch

from tichu_eval.full_position_pool import generate_full_position_pool
from tichu_training.cli.train_cotrain import _build_models
from tichu_training.featurizer import FEATURIZER_OUTPUT_DIM

_ARCH = {
    "model": dict(skill_dim=8, trunk_hidden=32, trunk_depth=1, trunk_out_dim=16, head_hidden=16),
    "schupfen_model": {"skill_dim": 8, "hidden": 16},
    "call_model": {"skill_dim": 8, "hidden": 16},
}


def _models():
    torch.manual_seed(0)
    return _build_models(_ARCH)


def _first_view(models, seat=0):
    """A realistic post-Schupfen Play PrivateState for `seat` — the root pMCPA
    adapts from. Played out with the frozen tiny nets via the blunder-miner
    recorder, then the seat's first Play Decision state is taken."""
    from tichu_inference.ml_agent import MLAgent
    from tichu_training.search.blunder_miner import record_round

    agent = MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"],
        tichu_call=models["tichu"], grand_call=models["grand"],
        skill_decile=9, partner_trick_guard=False,
    )
    positions = generate_full_position_pool(seed=31, n=1)
    _, decisions = record_round([agent] * 4, positions[0])
    d = next(x for x in decisions if x.seat == seat)
    return d.state.private_view(seat)


def _rows(models, *, seat=0, worlds=3, seed=5, **kwargs):
    from tichu_training.ppo.pmcpa import collect_pmcpa_rows

    return collect_pmcpa_rows(
        models, _first_view(models, seat), worlds=worlds, decisions_per_world=3,
        branches=3, skill_decile=9, seed=seed, **kwargs,
    )


def _agent(models, **kwargs):
    from tichu_training.ppo.pmcpa import PMCPAAgent

    return PMCPAAgent(
        models, skill_decile=9, worlds=3, decisions_per_world=3, branches=3,
        steps=3, lr=0.05, kl_coef=0.01, seed=0, **kwargs,
    )


def _plain_agent(models):
    from tichu_inference.ml_agent import MLAgent

    return MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"],
        tichu_call=models["tichu"], grand_call=models["grand"], skill_decile=9,
    )


def test_pmcpa_rows_are_valid_play_transitions():
    models = _models()
    rows = _rows(models)
    assert len(rows) > 0
    for r in rows:
        assert r["features"].shape == (FEATURIZER_OUTPUT_DIM,)
        assert bool(r["mask"][r["action"]])  # the taken action is legal
        assert r["old_logp"] <= 0.0
        assert abs(r["advantage"]) < 1000.0


def test_record_from_records_only_the_actor():
    # pMCPA adapts ONE seat's play head, so the gradient must see only that seat's
    # Play Decisions (the key divergence from vine, which records every seat).
    import random

    from tichu_inference.ml_agent import MLAgent
    from tichu_training.ppo.pmcpa import record_from
    from tichu_training.search.determinize import sample_determinized_world

    models = _models()
    agent = MLAgent.from_loaded(
        models["play"], schupfen=models["schupfen"],
        tichu_call=models["tichu"], grand_call=models["grand"],
        skill_decile=9, partner_trick_guard=False,
    )
    view = _first_view(models, seat=0)
    world = sample_determinized_world(view, None, random.Random(1))
    _, decisions = record_from([agent] * 4, world, actor_seat=0)
    assert len(decisions) > 0
    assert all(d.seat == 0 for d in decisions)


def test_pmcpa_rows_are_deterministic_per_seed():
    a = _rows(_models(), seed=5)
    b = _rows(_models(), seed=5)
    assert len(a) == len(b)
    assert [r["action"] for r in a] == [r["action"] for r in b]
    assert [r["advantage"] for r in a] == [r["advantage"] for r in b]


def test_sampled_worlds_preserve_hand_and_partition_unseen():
    # Danger (b) integrity: a sampled world keeps the actor's Hand exactly,
    # respects every opponent's hand size, and partitions the Unseen Cards with
    # no card duplicated or dropped.
    from tichu_engine.deck import fresh_deck
    from tichu_training.ppo.pmcpa import sample_worlds

    models = _models()
    view = _first_view(models, seat=0)
    worlds = sample_worlds(view, worlds=4, seed=3)
    assert len(worlds) == 4
    for w in worlds:
        assert w.hands[0] == view.hand  # actor's Hand untouched
        for seat in range(4):
            assert len(w.hands[seat]) == view.public.hand_sizes[seat]
        all_cards = set().union(*w.hands)
        played = set(view.public.played_cards_this_round)
        assert all_cards | played == set(fresh_deck())  # nothing lost
        assert len(all_cards) == sum(len(h) for h in w.hands)  # no duplicates


def _logp(net, row):
    feats = torch.as_tensor(row["features"], dtype=torch.float32).unsqueeze(0)
    skill = torch.tensor([9], dtype=torch.long)
    mask = torch.as_tensor(row["mask"], dtype=torch.bool)
    with torch.no_grad():
        logits = net(feats, skill)["play"][0].masked_fill(~mask, float("-inf"))
        return float(torch.log_softmax(logits, dim=0)[row["action"]])


def test_adaptation_moves_logp_toward_positive_advantage_actions():
    # The mechanism: with the anchor off, the play net moves toward the
    # higher-advantage actions the paired estimator found.
    models = _models()
    rows = _rows(models, emit_branches=True)
    positive = [r for r in rows if r["advantage"] > 0]
    assert positive  # branch emission surfaced better-than-baseline actions
    play_o = models["play"]
    play_a = copy.deepcopy(play_o)
    from tichu_training.ppo.pmcpa import adapt_play_net

    adapt_play_net(play_a, play_o, rows, steps=8, lr=0.05, kl_coef=0.0, skill_decile=9)
    # Advantage-weighted mean logp over the positive rows is the principled
    # measure of "moved toward the higher-advantage actions": the update
    # concentrates mass on the top-advantage actions, so a plain unweighted mean
    # is a fragile proxy (lower-but-positive rows can dip and drag it down even
    # when the mechanism worked — the seeded init at featurizer v6 exposed this).
    w = sum(r["advantage"] for r in positive)
    before = sum(r["advantage"] * _logp(play_o, r) for r in positive) / w
    after = sum(r["advantage"] * _logp(play_a, r) for r in positive) / w
    assert after > before


def test_kl_anchor_bounds_the_adaptation():
    # A large KL-anchor coefficient keeps theta_a close to theta_o — the in-Round
    # containment that stops a biased-sample gradient from cratering the Round.
    models = _models()
    rows = _rows(models, emit_branches=True)
    play_o = models["play"]
    from tichu_training.ppo.pmcpa import adapt_play_net

    free = copy.deepcopy(play_o)
    adapt_play_net(free, play_o, rows, steps=8, lr=0.05, kl_coef=0.0, skill_decile=9)
    anchored = copy.deepcopy(play_o)
    adapt_play_net(anchored, play_o, rows, steps=8, lr=0.05, kl_coef=1e4, skill_decile=9)
    drift = lambda net: sum(abs(_logp(net, r) - _logp(play_o, r)) for r in rows)
    assert drift(anchored) < drift(free)


def test_reset_restores_theta_o_bit_identical_and_never_mutates_it():
    # The "never persists" discipline: adaptation must not touch theta_o, and a
    # restore must return theta_a to theta_o byte-for-byte.
    from tichu_training.ppo.pmcpa import PMCPAAgent

    models = _models()
    theta_o_snapshot = {k: v.clone() for k, v in models["play"].state_dict().items()}
    agent = _agent(models)
    agent.on_round_start(_first_view(models, seat=0))
    # theta_o (the frozen offline net) is untouched by adaptation.
    for k, v in models["play"].state_dict().items():
        assert torch.equal(v, theta_o_snapshot[k])
    # theta_a diverged, then restore makes it bit-identical to theta_o again.
    agent.restore()
    for k, v in agent._seat_play[0].state_dict().items():
        assert torch.equal(v, theta_o_snapshot[k])


def test_one_instance_adapts_team_seats_independently():
    # The Tournament places one instance at both team-0 seats (a, b, a, b); each
    # seat must adapt on its OWN Hand, not clobber the other (the stateful-reuse
    # bug a shared theta_a would cause).
    from tichu_eval.play_full import play_full_round

    models = _models()
    agent = _agent(models)
    other = _plain_agent(models)
    positions = generate_full_position_pool(seed=31, n=1)
    pos = positions[0]
    play_full_round([agent, other, agent, other], pos.state, pos.grand_prefixes)
    assert 0 in agent._seat_play and 2 in agent._seat_play  # both seats adapted
    # The two seats hold independent theta_a (different Hands -> different rows).
    diff = any(
        not torch.equal(a, b)
        for a, b in zip(agent._seat_play[0].state_dict().values(),
                        agent._seat_play[2].state_dict().values())
    )
    assert diff


def test_play_full_round_fires_on_round_start_and_leaves_theta_o_frozen():
    from tichu_eval.play_full import play_full_round
    from tichu_training.ppo.pmcpa import PMCPAAgent

    models = _models()
    theta_o_snapshot = {k: v.clone() for k, v in models["play"].state_dict().items()}
    agent = _agent(models)
    others = _plain_agent(models)
    positions = generate_full_position_pool(seed=31, n=1)
    pos = positions[0]
    # Seat 0 is the adapting agent; the rest are plain theta_o MLAgents.
    play_full_round([agent, others, others, others], pos.state, pos.grand_prefixes)
    # The adapting agent's seat-0 theta_a diverged (adaptation ran via the hook)...
    assert 0 in agent._seat_play
    diverged = any(
        not torch.equal(a, b)
        for a, b in zip(agent._seat_play[0].state_dict().values(),
                        models["play"].state_dict().values())
    )
    assert diverged
    # ...and theta_o stayed frozen throughout the Round.
    for k, v in models["play"].state_dict().items():
        assert torch.equal(v, theta_o_snapshot[k])


def test_oracle_agent_adapts_on_the_true_world_via_oracle_hook():
    # Diagnostic ceiling variant (ADR-0036): the runner's oracle hook hands it the
    # full GameState, so it adapts on the TRUE world (worlds_override) — theta_a
    # diverges, theta_o stays frozen, and the normal sampling hook is bypassed.
    from tichu_eval.play_full import play_full_round
    from tichu_training.ppo.pmcpa import PMCPAOracleAgent

    models = _models()
    theta_o_snapshot = {k: v.clone() for k, v in models["play"].state_dict().items()}
    agent = PMCPAOracleAgent(
        models, skill_decile=9, worlds=1, decisions_per_world=6, branches=3,
        steps=3, lr=0.01, kl_coef=1.0, seed=0,
    )
    assert hasattr(agent, "on_round_start_oracle")
    other = _plain_agent(models)
    pos = generate_full_position_pool(seed=31, n=1)[0]
    play_full_round([agent, other, agent, other], pos.state, pos.grand_prefixes)
    assert 0 in agent._seat_play  # the oracle hook fired and adapted seat 0
    diverged = any(
        not torch.equal(a, b)
        for a, b in zip(agent._seat_play[0].state_dict().values(),
                        models["play"].state_dict().values())
    )
    assert diverged
    for k, v in models["play"].state_dict().items():
        assert torch.equal(v, theta_o_snapshot[k])  # theta_o never mutated


def test_play_full_round_unchanged_without_the_hook():
    # Agents lacking on_round_start (every existing Agent) take the byte-identical
    # path — the hook is additive.
    from tichu_eval.play_full import play_full_round

    models = _models()
    agent = _plain_agent(models)
    positions = generate_full_position_pool(seed=31, n=1)
    pos = positions[0]
    r = play_full_round([agent, agent, agent, agent], pos.state, pos.grand_prefixes)
    assert isinstance(r.total, tuple) and len(r.total) == 2
