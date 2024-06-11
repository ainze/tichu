import unittest
from game import Game
from cards import Card
from player import RandomPlayer


class TestGame(unittest.TestCase):
    def setUp(self):
        self.game = Game([RandomPlayer("P1"), RandomPlayer("P2"), RandomPlayer("P3"), RandomPlayer("P4")])

    def test_is_bomb_same_rank(self):
        # Test four cards of the same rank
        bomb = [
            Card('Hearts', '10'),
            Card('Spades', '10'),
            Card('Diamonds', '10'),
            Card('Clubs', '10')
        ]
        self.assertTrue(self.game.is_bomb(bomb))

    def test_is_not_bomb_different_ranks(self):
        # Test four cards of different ranks
        not_bomb = [
            Card('Hearts', '10'),
            Card('Spades', 'J'),
            Card('Diamonds', '10'),
            Card('Clubs', '10')
        ]
        self.assertFalse(self.game.is_bomb(not_bomb))

    def test_is_bomb_straight_flush(self):
        # Test a valid straight flush
        straight_flush = [
            Card('Hearts', '9'),
            Card('Hearts', '10'),
            Card('Hearts', 'J'),
            Card('Hearts', 'Q'),
            Card('Hearts', 'K')
        ]
        self.assertTrue(self.game.is_bomb(straight_flush))

    def test_is_not_bomb_not_straight_flush(self):
        # Test an invalid straight flush
        not_straight_flush = [
            Card('Hearts', '9'),
            Card('Hearts', '10'),
            Card('Hearts', 'J'),
            Card('Hearts', 'Q'),
            Card('Clubs', 'K')
        ]
        self.assertFalse(self.game.is_bomb(not_straight_flush))

    def test_is_not_bomb_too_few_cards(self):
        # Test too few cards to form a bomb
        too_few_cards = [
            Card('Hearts', '10'),
            Card('Spades', '10'),
            Card('Diamonds', '10')
        ]
        self.assertFalse(self.game.is_bomb(too_few_cards))

if __name__ == "__main__":
    unittest.main()