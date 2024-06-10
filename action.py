from cards import Card


class Action:
    def __init__(self, cards: [Card], action: str):
        self.cards = cards
        self.action = action

