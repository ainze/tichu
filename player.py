import random
from abc import ABC
from typing import List, Union

from action import Action
from cards import Card, DogCard, DragonCard, PhoenixCard, MahJongCard
from game import Game


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
        # Check if the player has the cards they want to play
        for card in new_trick:
            if card not in cards_available:
                return False

        if len(last_trick) == 0:
            return True # any card is valid if this is the first hand to play
        if Game.is_bomb(new_trick):
            if not Game.is_bomb(last_trick):
                return True #bombs can always be played
            else:
                return Game.compare_bombs(new_trick, last_trick)

        # If the last trick is a bomb and the new trick is not, the new trick is invalid
        if Game.is_bomb(last_trick) and not Game.is_bomb(new_trick):
            return False

        # Check if the new trick matches the type and is higher in value than the last trick
        return Game.compare_tricks(new_trick, last_trick)




        if(len(new_trick) == len(last_trick))

=        if (len(last_trick))



class RandomPlayer(Player):

    def ask_trick(self, cards_on_table: List[List[Card]]) -> Action:
        card = random.choice(self.cards)
        self.cards.remove(card)

        return Action([card], False)

    def ask_grandTichu(self) -> bool:
        return random.choice([True, False])


