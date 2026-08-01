"""Self-play-derived Belief-example emit (ADR-0041).

Sibling of [emit.py](emit.py), which reads labels off a parsed BSW replay. This
path reads them off the live engine `GameState` as a **Round** is played, so a
**Belief Model** can be trained on the *self-play* distribution it is consumed on
— removing the human-corpus/self-play skew, and needing no parse pass.

At **Featurizer v6** the policy Feature Vector already carries ADR-0028's B-core
History channels (`declined_top` / `lead_summary` / `pass_pressure`), so the
belief input here is the Feature Vector itself — see `belief/input_spec.py`.
"""

from __future__ import annotations

import numpy as np

from tichu_training.belief.dataset import BeliefExample
from tichu_training.featurizer import FEATURIZER_VERSION, featurize

_NUM_OPPONENTS = 3
_NUM_CARDS = 56


def belief_examples_for_selfplay_round(
    agents,
    position,
    *,
    game_id: int = 0,
    round_id: int = 0,
    rich_history: bool = False,
) -> list[BeliefExample]:
    """One `BeliefExample` per Play Decision of a self-played Round, from the
    acting seat's view. Labels are the three opponents' true Hands in
    relative-seat order (next / partner / previous); the mask is the **Unseen
    Cards** — read from the same `GameState` that produces the features, so they
    are always fresh to the featurise moment.

    `rich_history=True` appends `RichHistory`'s recovered channels (ADR-0041
    follow-up) — decline *context*, combination length, declined Bombs, Trick
    stakes, and the Mahjong-wish void — which v6's 27-dim B-core block discards.
    Strictly additive: the v6 prefix is byte-identical either way.
    """
    from tichu_eval.play_full import play_full_round

    from tichu_training.belief.rich_history import RichHistory

    out: list[BeliefExample] = []
    history = RichHistory() if rich_history else None

    def observer(seat, game_state, action):
        # History carries Decisions made BEFORE this one — the acting seat's own
        # action is folded in afterwards, mirroring `emit.belief_examples_for_round`.
        out.append(_example(
            game_state, seat, game_id=game_id, round_id=round_id, history=history,
        ))
        if history is not None:
            history.update(seat, action, game_state)

    play_full_round(
        agents, position.state, position.grand_prefixes, state_observer=observer
    )
    return out


def belief_examples_for_selfplay_rounds(
    agents_factory,
    positions,
    *,
    game_id: int = 0,
    rich_history: bool = False,
) -> list[BeliefExample]:
    """Emit over many Starting Positions, tagging each example with the index of
    the Round it came from. `agents_factory()` builds a fresh seat list per Round
    so nothing carries across deals."""
    out: list[BeliefExample] = []
    for round_id, position in enumerate(positions):
        out.extend(
            belief_examples_for_selfplay_round(
                agents_factory(), position, game_id=game_id, round_id=round_id,
                rich_history=rich_history,
            )
        )
    return out


def round_level_split(
    examples: list[BeliefExample],
    *,
    holdout_frac: float,
    seed: int,
) -> tuple[list[BeliefExample], list[BeliefExample]]:
    """Split `(train, holdout)` by **whole Round**, never by example.

    Every hidden-hand configuration is round-unique, so an example-level split
    puts the same deal on both sides and the holdout stops measuring
    generalisation — the memorisation mode the ADR-0033 post-mortem identified.
    """
    import random as _random

    keys = sorted({(ex.game_id, ex.round_id) for ex in examples})
    rng = _random.Random(seed)
    rng.shuffle(keys)
    n_holdout = int(round(len(keys) * holdout_frac))
    held = set(keys[:n_holdout])
    train = [ex for ex in examples if (ex.game_id, ex.round_id) not in held]
    holdout = [ex for ex in examples if (ex.game_id, ex.round_id) in held]
    return train, holdout


def card_counting_marginals(private_state) -> np.ndarray:
    """`(3, 56)` occupancy probabilities from **public information only** — the
    max-entropy predictor that knows opponent `hand_sizes` and nothing else.

    Every **Unseen Card** is split across the three opponents (relative-seat
    order: next / partner / previous) in proportion to their hand sizes; cards
    the actor can see — own **Hand** or already played — carry zero mass. This is
    the floor a trained **Belief Model** must beat, and the belief-off marginals
    for the **Belief-Optimal Chooser** (ADR-0041).
    """
    from tichu_training.card_slots import card_slot

    root = private_state.player
    rel_seats = ((root + 1) % 4, (root + 2) % 4, (root + 3) % 4)
    sizes = np.array(
        [private_state.public.hand_sizes[s] for s in rel_seats], dtype=np.float32
    )
    total_unseen = float(sizes.sum())

    probs = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=np.float32)
    if total_unseen <= 0:
        return probs

    seen = {card_slot(c) for c in private_state.hand}
    seen |= {card_slot(c) for c in private_state.public.played_cards_this_round}
    unseen = [s for s in range(_NUM_CARDS) if s not in seen]
    probs[:, unseen] = (sizes / total_unseen)[:, None]
    return probs


def which_opponent_top1(probs: np.ndarray, example: BeliefExample) -> tuple[int, int]:
    """`(hits, cards)` — for each **Unseen Card**, whether `argmax` over the three
    opponents picks the one who actually holds it.

    The discriminating belief metric (chance ≈ 1/3): masked BCE and accuracy@0.5
    are both dominated by *whether* a card is unknown, which every predictor knows.
    Only Unseen Cards enter the denominator, so nothing is credited for
    common knowledge.
    """
    card_mask = np.asarray(example.mask)[0].astype(bool)
    cards = int(card_mask.sum())
    if cards == 0:
        return 0, 0
    picked = np.asarray(probs)[:, card_mask].argmax(axis=0)
    truth = np.asarray(example.labels)[:, card_mask].argmax(axis=0)
    return int((picked == truth).sum()), cards


def train_selfplay_belief(
    examples: list[BeliefExample],
    *,
    holdout: list[BeliefExample] | None = None,
    hidden: int = 256,
    depth: int = 4,
    trunk_out_dim: int | None = None,
    residual: bool = False,
    holder_loss: bool = False,
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 1.0e-3,
    seed: int = 0,
):
    """Fit a **Belief Model** on self-play examples.

    Returns the model alone when `holdout` is None; `(model, history)` otherwise,
    where the model is the **best held-out epoch**, not the last.

    Early stopping is not optional here. Every Round is a unique 4-hand deal, so
    the hidden-hand target is round-unique and a fit left to run **memorises
    deals** — measured at train top-1 0.998 with held-out top-1 *below* the
    card-counting floor. That is the same mechanism the ADR-0033 post-mortem
    identified for perfect-info features under a round-level split.

    Deliberately small and local: the `cli/train_belief` path is built around
    materialised BSW bundles and the v5 tier prefixes, which no longer line up
    with Featurizer v6.
    """
    import copy

    import torch

    from tichu_training.belief.model import (
        BeliefModel, belief_holder_loss, belief_loss,
    )

    torch.manual_seed(seed)
    feats = torch.from_numpy(np.stack([ex.features for ex in examples]))
    labels = torch.from_numpy(np.stack([ex.labels for ex in examples]))
    mask = torch.from_numpy(np.stack([np.asarray(ex.mask) for ex in examples]))

    model = BeliefModel(
        feature_dim=feats.shape[1], hidden=hidden, depth=depth,
        trunk_out_dim=trunk_out_dim, residual=residual,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    history: list[dict] = []
    best_state, best_score = None, None
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(examples), generator=gen)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            opt.zero_grad()
            logits = model(feats[rows])
            loss = (
                belief_holder_loss(logits, labels[rows], mask[rows][:, 0])
                if holder_loss
                else belief_loss(logits, labels[rows], mask[rows])
            )
            loss.backward()
            opt.step()
        if holdout is None:
            continue
        model.eval()
        score = score_belief_top1(model, holdout, holder_loss=holder_loss)
        history.append({"epoch": epoch, "holdout_top1": score})
        if best_score is None or score > best_score:
            best_state, best_score = copy.deepcopy(model.state_dict()), score

    model.eval()
    if holdout is None:
        return model
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def belief_probabilities(model, features, *, holder_loss: bool = False):
    """The `(N, 3, 56)` marginals `model` implies, under its own parameterisation:
    per-card `softmax` over the three opponents when it was fit with
    `belief_holder_loss`, independent `sigmoid` otherwise."""
    import torch

    from tichu_training.belief.model import holder_probabilities

    with torch.no_grad():
        logits = model(features)
        activated = (
            holder_probabilities(logits) if holder_loss else torch.sigmoid(logits)
        )
    return activated.numpy()


def score_belief_top1(
    model, examples: list[BeliefExample], *, holder_loss: bool = False
) -> float:
    """Which-opponent top-1 of `model` over `examples` (see `which_opponent_top1`)."""
    import torch

    feats = torch.from_numpy(np.stack([ex.features for ex in examples]))
    probs = belief_probabilities(model, feats, holder_loss=holder_loss)
    hits = cards = 0
    for row, ex in enumerate(examples):
        h, c = which_opponent_top1(probs[row], ex)
        hits += h
        cards += c
    return hits / cards if cards else 0.0


def save_examples(path, examples: list[BeliefExample]) -> None:
    """Persist emitted examples so capacity sweeps re-read instead of re-emitting
    (emit is minutes, training is seconds). The mask is derivable — a card is
    unknown iff some opponent holds it — so it is not stored."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.stack([ex.features for ex in examples]),
        labels=np.stack([ex.labels for ex in examples]),
        cards_played=np.array([ex.cards_played for ex in examples], dtype=np.int32),
        game_id=np.array([ex.game_id for ex in examples], dtype=np.int32),
        round_id=np.array([ex.round_id for ex in examples], dtype=np.int32),
    )


def load_examples(path) -> list[BeliefExample] | None:
    """Read back `save_examples`, or None if the file does not exist.

    Each stored array is decompressed **once**: `NpzFile.__getitem__` re-reads the
    whole array on every access, so indexing it inside the per-example loop turns
    one read into one-per-example and exhausts memory.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as handle:
        features = handle["features"]
        labels = handle["labels"]
        cards_played = handle["cards_played"]
        game_id = handle["game_id"]
        round_id = handle["round_id"]
    card_mask = labels.any(axis=1)
    return [
        BeliefExample(
            features=features[i],
            labels=labels[i],
            mask=np.broadcast_to(card_mask[i], (_NUM_OPPONENTS, _NUM_CARDS)).copy(),
            cards_played=int(cards_played[i]),
            game_id=int(game_id[i]),
            round_id=int(round_id[i]),
        )
        for i in range(len(labels))
    ]


def save_belief_checkpoint(path, model, *, holder_loss: bool = False) -> None:
    """Write a fitted **Belief Model** with the shape needed to rebuild it."""
    from pathlib import Path

    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": _belief_in_features(model),
            "hidden": _belief_hidden(model),
            "depth": len(model.trunk.blocks) if model.residual else 0,
            "trunk_out_dim": (model.trunk.out_dim if model.residual else None),
            "residual": bool(model.residual),
            "holder_loss": bool(holder_loss),
            "featurizer_version": FEATURIZER_VERSION,
        },
        path,
    )


def load_belief_marginals_fn(path, *, history_provider=None):
    """`marginals_fn(private_state) -> (3, 56)` for the **belief-on** arm of the
    **Belief-Optimal Chooser**.

    Relative-seat order (next / partner / previous) matches `card_counting_
    marginals` and `belief.emit`, so the two arms sample from identically-ordered
    worlds. The featurizer version is pinned: a v5 checkpoint against v6 features
    is silent garbage, which is how both on-disk belief checkpoints died.

    `history_provider(seat) -> np.ndarray` supplies a `RichHistory` block for
    checkpoints fit with `rich_history=True`. A Chooser sees only its own turns,
    so the caller — which walks the whole Round — owns the accumulator. The input
    width is checked against the checkpoint, so a missing provider raises instead
    of feeding the model a truncated vector.
    """
    import torch

    from tichu_training.belief.model import BeliefModel

    blob = torch.load(path, map_location="cpu", weights_only=False)
    stamped = blob.get("featurizer_version")
    if stamped is not None and stamped != FEATURIZER_VERSION:
        raise ValueError(
            f"belief checkpoint is featurizer {stamped}, loader expects "
            f"{FEATURIZER_VERSION}"
        )
    model = BeliefModel(
        feature_dim=int(blob["feature_dim"]),
        hidden=int(blob["hidden"]),
        depth=int(blob.get("depth") or 4),
        trunk_out_dim=blob.get("trunk_out_dim"),
        residual=bool(blob.get("residual", False)),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    holder_loss = bool(blob.get("holder_loss", False))

    expected = int(blob["feature_dim"])

    def marginals_fn(private_state) -> np.ndarray:
        vec = featurize(private_state)
        if history_provider is not None:
            vec = np.concatenate([vec, history_provider(private_state.player)])
        if vec.shape[0] != expected:
            raise ValueError(
                f"belief checkpoint expects {expected} input dims, got "
                f"{vec.shape[0]} — pass history_provider= for a rich-history "
                f"checkpoint (or drop it for a plain one)"
            )
        features = torch.from_numpy(vec.astype(np.float32)).unsqueeze(0)
        probs = belief_probabilities(model, features, holder_loss=holder_loss)
        return probs[0].astype(np.float32)

    return marginals_fn


def _belief_in_features(model) -> int:
    return int(
        model.trunk.input_proj.in_features if model.residual else model.fc1.in_features
    )


def _belief_hidden(model) -> int:
    return int(
        model.trunk.input_proj.out_features
        if model.residual
        else model.fc1.out_features
    )


def floor_marginals_for_example(example: BeliefExample) -> np.ndarray:
    """The card-counting floor for a scored `BeliefExample`.

    Same predictor as `card_counting_marginals`, reached from the other side: that
    one reads `hand_sizes` off a live `PrivateState` (what the **Belief-Optimal
    Chooser** consumes), this one recovers them from the example's own labels
    (what scoring consumes). Their equivalence is pinned by test.
    """
    card_mask = np.asarray(example.mask)[0].astype(bool)
    unseen = int(card_mask.sum())
    probs = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=np.float32)
    if unseen:
        sizes = np.asarray(example.labels)[:, card_mask].sum(axis=1) / unseen
        probs[:, card_mask] = sizes[:, None]
    return probs


def masked_bce(probs: np.ndarray, example: BeliefExample) -> tuple[float, int]:
    """`(summed binary cross-entropy, positions)` over masked-in positions.

    The calibration companion to `which_opponent_top1`: the **Determinized World**
    sampler consumes full marginals, so two predictors can agree on every argmax
    and still induce different world distributions. Every `(opponent, Unseen
    Card)` position is scored; positions the actor can see are excluded, so no
    predictor is rewarded for confidence about common knowledge.
    """
    mask = np.asarray(example.mask).astype(bool)
    if not mask.any():
        return 0.0, 0
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y = np.asarray(example.labels, dtype=np.float64)
    per_pos = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(per_pos[mask].sum()), int(mask.sum())


def holder_nll(probs: np.ndarray, example: BeliefExample) -> tuple[float, int]:
    """`(summed negative log-likelihood of the true holder, Unseen Cards)`.

    The calibration metric that matches what the sampler *does*: `_draw_holder`
    normalises its weights across the eligible opponents, so only the **relative**
    mass per card governs which world comes out — not the absolute values
    `masked_bce` scores. Comparable between the independent-sigmoid and the
    per-card-softmax parameterisations, and `ln(3)` is the uninformative baseline.
    """
    card_mask = np.asarray(example.mask)[0].astype(bool)
    cards = int(card_mask.sum())
    if cards == 0:
        return 0.0, 0
    weights = np.asarray(probs, dtype=np.float64)[:, card_mask]
    totals = weights.sum(axis=0, keepdims=True)
    # A predictor that puts no mass on a card says nothing about its holder.
    safe = np.where(totals > 0, weights / np.where(totals > 0, totals, 1.0), 1.0 / 3.0)
    truth = np.asarray(example.labels)[:, card_mask].argmax(axis=0)
    picked = safe[truth, np.arange(cards)]
    return float(-np.log(np.clip(picked, 1e-12, None)).sum()), cards


def score_floor_top1(examples: list[BeliefExample]) -> float:
    """Which-opponent top-1 of the card-counting floor over `examples`."""
    hits = cards = 0
    for ex in examples:
        h, c = which_opponent_top1(floor_marginals_for_example(ex), ex)
        hits += h
        cards += c
    return hits / cards if cards else 0.0


def _example(
    game_state, seat: int, *, game_id: int, round_id: int, history=None,
) -> BeliefExample:
    from tichu_training.card_slots import card_slot

    hands = game_state.hands
    rel_seats = ((seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4)
    labels = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=np.float32)
    for opp_idx, opp_seat in enumerate(rel_seats):
        for card in hands[opp_seat]:
            labels[opp_idx, card_slot(card)] = 1.0
    # A card is "unknown" iff some opponent holds it (≡ not own, not played).
    card_mask = labels.any(axis=0)
    features = featurize(game_state.private_view(seat))
    if history is not None:
        features = np.concatenate([features, history.block(seat)]).astype(np.float32)
    return BeliefExample(
        features=features,
        labels=labels,
        mask=np.broadcast_to(card_mask, (_NUM_OPPONENTS, _NUM_CARDS)).copy(),
        cards_played=_NUM_CARDS - sum(len(h) for h in hands),
        game_id=game_id,
        round_id=round_id,
    )
