import logging
import random
from abc import ABC
from typing import List, Union

from GameRules import GameRules
from action import Action
from cards import Card, DogCard, DragonCard, PhoenixCard, MahJongCard


class Player(ABC):

    def __init__(self, name: str):
        self.name = name
        self.cards: List[Card] = []
        self.stored_cards: List[Card] = []

    def __repr__(self):
        return f"Player(name={self.name}, cards={self.cards}, stored_cards={self.stored_cards})"

    def ask_trick(self, cards_on_table: List[List[Card]]) -> Action:
        raise NotImplementedError()

    def ask_grandTichu(self) -> bool:
        raise NotImplementedError()

    def rec_cards(self, cardsRec):
        self.cards.extend(cardsRec)

    def hasMahJong(self) -> bool:
        return any(card.rank == 'MahJong' for card in self.cards)

    def store_cards(self, store_cards):
        self.stored_cards.extend(store_cards)

    def hasCards(self) -> bool:
        return len(self.cards) > 0

    def hasNoCards(self) -> bool:
        return not self.hasCards()


class RandomPlayer(Player):

    # def ask_trick(self, cards_on_table: List[List[Card]]) -> Action:
    #
    #     for card in self.cards:
    #         if GameRules.is_valid_trick(cards_on_table, [card], self.cards):
    #             self.cards.remove(card)
    #             return Action([card], 'PLAY')
    #     return Action([], 'PASS')

    def ask_trick(self, cards_on_table: List[List[Card]]) -> Action:
        logging.info(f'Player {self.name} ask_trick')
        possible_plays = self.generate_possible_plays()
        random.shuffle(possible_plays)  # we are a random player are we not
        for play in possible_plays:
            if GameRules.is_valid_trick(cards_on_table, play, self.cards):
                for card in play:
                    self.cards.remove(card)
                return Action(play, 'PLAY')
        return Action([], 'PASS')

    def ask_grandTichu(self) -> bool:
        return random.choice([True, False])

    def generate_possible_plays(self) -> List[List[Card]]:
        # Todo: it is much better to pass the table, less options to generate
        plays = []
        # Generate singles
        plays.extend([[card] for card in self.cards])

        # Generate pairs
        plays.extend(
            [[self.cards[i], self.cards[j]] for i in range(len(self.cards)) for j in range(i + 1, len(self.cards)) if
             self.cards[i].rank == self.cards[j].rank])

        # Generate triples
        plays.extend([[self.cards[i], self.cards[j], self.cards[k]] for i in range(len(self.cards)) for j in
                      range(i + 1, len(self.cards)) for k in range(j + 1, len(self.cards)) if
                      self.cards[i].rank == self.cards[j].rank == self.cards[k].rank])

        # Generate full houses
        triples = [[self.cards[i], self.cards[j], self.cards[k]] for i in range(len(self.cards)) for j in
                   range(i + 1, len(self.cards)) for k in range(j + 1, len(self.cards)) if
                   self.cards[i].rank == self.cards[j].rank == self.cards[k].rank]
        pairs = [[self.cards[i], self.cards[j]] for i in range(len(self.cards)) for j in range(i + 1, len(self.cards))
                 if self.cards[i].rank == self.cards[j].rank]
        for triple in triples:
            remaining_cards = [card for card in self.cards if card not in triple]
            full_houses = [[triple[0], triple[1], triple[2], pair[0], pair[1]] for pair in pairs if
                           pair[0] in remaining_cards and pair[1] in remaining_cards]
            plays.extend(full_houses)

        # Generate straights (assuming 5-card straights only for simplicity)
        sorted_cards = sorted(self.cards, key=lambda card: (
        Card.RANKS_EXTENDED.index(card.rank), Card.SUITS_EXTENDED.index(card.suit)))
        for i in range(len(sorted_cards) - 4):
            straight = sorted_cards[i:i + 5]
            if all(Card.RANKS_EXTENDED.index(straight[j].rank) + 1 == Card.RANKS_EXTENDED.index(straight[j + 1].rank)
                   for j in range(4)):
                plays.append(straight)

        # Generate bombs (four-of-a-kind)
        bombs = [[self.cards[i], self.cards[j], self.cards[k], self.cards[l]] for i in range(len(self.cards)) for j
                 in range(i + 1, len(self.cards)) for k in range(j + 1, len(self.cards)) for l in
                 range(k + 1, len(self.cards)) if
                 self.cards[i].rank == self.cards[j].rank == self.cards[k].rank == self.cards[l].rank]
        plays.extend(bombs)

        # Generate straight flush bombs
        for suit in Card.SUITS:
            suited_cards = [card for card in sorted_cards if card.suit == suit]
            for i in range(len(suited_cards) - 4):
                straight_flush = suited_cards[i:i + 5]
                if all(Card.RANKS.index(straight_flush[j].rank) + 1 == Card.RANKS.index(straight_flush[j + 1].rank)
                       for j in range(4)):
                    plays.append(straight_flush)

        return plays
