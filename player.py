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




class RandomPlayer(Player):

    def ask_trick(self, cards_on_table: List[List[Card]]) -> Action:

        for card in self.cards:
            if GameRules.is_valid_trick(cards_on_table, [card], self.cards):
                self.cards.remove(card)
                return Action([card], 'PLAY')
        return Action([], 'PASS')


    def ask_grandTichu(self) -> bool:
        return random.choice([True, False])


