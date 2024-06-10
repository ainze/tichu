import random
from typing import List, Union


class Card:
    """
        Represents a playing card in the Tishu game.

        Attributes:
            suit (str): The suit of the card.
            rank (str): The rank of the card.
        """
    SUITS = ['Diamonds', 'Clubs', 'Hearts', 'Spades']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, suit: str, rank: str):
        if suit == 'Special':
            if rank not in ['Dog', 'Dragon', 'Phoenix', 'MahJong']:
                raise ValueError(f"Invalid Special rank: {rank}")
        elif suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}")
        elif rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}")
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

    def __lt__(self, other):
        if isinstance(other, (DogCard, DragonCard, PhoenixCard, MahJongCard)):
            return True
        if self.RANKS.index(self.rank) < other.RANKS.index(other.rank):
            return True
        if self.RANKS.index(self.rank) == other.RANKS.index(other.rank):
            return self.SUITS.index(self.suit) < self.SUITS.index(other.suit)
        return False

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit


class DogCard(Card):
    def __init__(self):
        super().__init__(suit="Special", rank="Dog")

    def __repr__(self):
        return "Dog"


class DragonCard(Card):
    def __init__(self):
        super().__init__(suit="Special", rank="Dragon")

    def __repr__(self):
        return "Dragon"


class PhoenixCard(Card):
    def __init__(self):
        super().__init__(suit="Special", rank="Phoenix")

    def __repr__(self):
        return "Phoenix"


class MahJongCard(Card):
    def __init__(self):
        super().__init__(suit="Special", rank="MahJong")

    def __repr__(self):
        return "MahJong"


class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Card.SUITS for rank in Card.RANKS]
        self.cards.extend([DogCard(), DragonCard(), PhoenixCard(), MahJongCard()])
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_hand(self, num_cards: int) -> List[Union[Card, DogCard, DragonCard, PhoenixCard, MahJongCard]]:
        if len(self.cards) < num_cards:
            raise ValueError("Not enough cards in the deck to deal the hand")
        hand = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return hand

    def __repr__(self):
        return f"Deck of {len(self.cards)} cards"


if __name__ == "__main__":
    deck = Deck()
    print("Shuffled Deck:", deck)
    hand = deck.deal_hand(5)
    print("Dealt Hand:", hand)
    print("Remaining Deck:", deck)