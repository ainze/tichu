import random
from abc import ABC
from typing import List, Union

from action import Action
from cards import Card, DogCard, DragonCard, PhoenixCard, MahJongCard


class Player(ABC):

    def __init__(self, name: str):
        self.name = name
        self.cards: List[Card] = []
        self.stored_cards: List[Card] = []

    def __repr__(self):
        return f"Player(name={self.name})"

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

    def is_valid_trick(self, last_trick: List[Card], new_trick: List[Card], cards_available: List[Card]) -> bool:
        if (len(last_trick) == 0):
            return True # any card is valid if this is the first hand to play
#         if(is_bomb(new_trick)):
#
#
#         if(len(new_trick) == len(last_trick))
#
# =        if (len(last_trick))



class RandomPlayer(Player):

    def ask_trick(self, cards_on_table: List[List[Card]]) -> Action:
        card = random.choice(self.cards)
        self.cards.remove(card)

        return Action([card], False)

    def ask_grandTichu(self) -> bool:
        return random.choice([True, False])


