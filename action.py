from typing import List

from cards import Card


class Action:
    def __init__(self, cards: List[Card], action: str):
        self.cards = cards
        self.action = action

