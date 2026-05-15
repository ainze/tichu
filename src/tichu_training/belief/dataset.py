"""Synthetic belief-model dataset.

Each example simulates a mid-round game state:
  * features: random public-history feature vector.
  * labels: (3 opponents, 56 cards) bool — which opponent holds which card.
  * mask:   (3, 56) bool — positions where the answer is unknown to the
            acting player (i.e. the card is somewhere among the opponents).

The card-distribution scheme: shuffle 56 card ids, take 14 for own hand,
then 14 for each of 3 opponents, the remainder are "played". Labels are
1 for the (opponent, card) of cards in that opponent's hand; mask is True
for all card columns NOT in own_hand and NOT already played.
"""

from dataclasses import dataclass

import numpy as np


_NUM_CARDS = 56
_NUM_OPPONENTS = 3


@dataclass(frozen=True)
class BeliefExample:
    features: np.ndarray   # (feature_dim,) float32
    labels: np.ndarray     # (3, 56) float32 — 1 where opp_i holds card_j
    mask: np.ndarray       # (3, 56) bool   — True where the answer is unknown


class SyntheticBeliefDataset:
    def __init__(self, *, seed: int, n_examples: int, feature_dim: int) -> None:
        self.seed = seed
        self.n_examples = n_examples
        self.feature_dim = feature_dim

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_examples):
            features = rng.standard_normal(self.feature_dim).astype(np.float32)
            card_ids = np.arange(_NUM_CARDS)
            rng.shuffle(card_ids)
            own_hand = set(card_ids[:14].tolist())
            opp_hands = [set(card_ids[14 + 14 * i : 14 + 14 * (i + 1)].tolist())
                         for i in range(_NUM_OPPONENTS)]
            played = set(card_ids[14 + 14 * _NUM_OPPONENTS:].tolist())

            labels = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=np.float32)
            mask = np.zeros((_NUM_OPPONENTS, _NUM_CARDS), dtype=bool)
            for opp_idx, hand in enumerate(opp_hands):
                for card_id in hand:
                    labels[opp_idx, card_id] = 1.0
            for card_id in range(_NUM_CARDS):
                if card_id in own_hand or card_id in played:
                    continue
                mask[:, card_id] = True

            yield BeliefExample(features=features, labels=labels, mask=mask)
