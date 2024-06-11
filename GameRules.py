from typing import List

from cards import Card


class GameRules:

    def __init__(self):
        self.rules = []

    @classmethod
    def is_valid_trick(cls, last_trick: List[Card], new_trick: List[Card], cards_available: List[Card]) -> bool:
        # Check if the player has the cards they want to play
        for card in new_trick:
            if card not in cards_available:
                return False

        if len(last_trick) == 0:
            return True  # any card is valid if this is the first hand to play
        if cls.is_bomb(new_trick):
            if not cls.is_bomb(last_trick):
                return True  # bombs can always be played
            else:
                return Game.compare_bombs(new_trick, last_trick)

        # If the last trick is a bomb and the new trick is not, the new trick is invalid
        if cls.is_bomb(last_trick) and not cls.is_bomb(new_trick):
            return False

        # Check if the new trick matches the type and is higher in value than the last trick
        return cls.compare_tricks(new_trick, last_trick)

    @classmethod
    def is_bomb(cls, new_trick: List[Card]) -> bool:
        if len(new_trick) == 4:
            # check if all 4 are same rank
            if any(card.rank != new_trick[0].rank for card in new_trick):
                return False  # this is not a bomb since rank is different
            else:
                return True
        elif len(new_trick) > 4:
            if any(card.suit != new_trick[0].suit for card in new_trick):
                return False  # cards need to be same suite for straight

            # Check if the ranks are in a climbing sequence
            # Convert ranks to their indices for comparison
            rank_indices = [Card.RANKS.index(card.rank) for card in new_trick]
            rank_indices.sort()

            for i in range(len(rank_indices) - 1):
                if rank_indices[i] + 1 != rank_indices[i + 1]:
                    return False  # Ranks are not in a climbing sequence
            return True
        else:
            return False

    @classmethod
    def compare_bombs(cls, new_bomb: List[Card], last_bomb: List[Card]) -> bool:
        # Assume both new_bomb and last_bomb are valid bombs
        # Compare based on rank if they are four-of-a-kind
        if len(new_bomb) == 4 and len(last_bomb) == 4:
            new_bomb_rank = Card.RANKS.index(new_bomb[0].rank)
            last_bomb_rank = Card.RANKS.index(last_bomb[0].rank)
            return new_bomb_rank > last_bomb_rank

        # If they are straight flushes, compare the highest card
        new_bomb_max_rank = max(Card.RANKS.index(card.rank) for card in new_bomb)
        last_bomb_max_rank = max(Card.RANKS.index(card.rank) for card in last_bomb)
        return new_bomb_max_rank > last_bomb_max_rank

    @classmethod
    def compare_tricks(cls, new_trick: List[Card], last_trick: List[Card]) -> bool:
        # Assume both new_trick and last_trick are of the same type
        if len(new_trick) != len(last_trick):
            return False

        new_trick_rank = [Card.RANKS.index(card.rank) for card in new_trick]
        last_trick_rank = [Card.RANKS.index(card.rank) for card in last_trick]

        new_trick_rank.sort()
        last_trick_rank.sort()

        return new_trick_rank > last_trick_rank
