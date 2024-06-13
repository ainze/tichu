import unittest

from GameRules import GameRules
from game import Game
from cards import Card
from player import RandomPlayer


class TestGameClassMethods(unittest.TestCase):
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


    def test_is_valid_trick_simple_one_card(self):
        last_trick = [Card('Hearts', '9')]
        new_trick = [Card('Hearts', '10')]
        cards_available = [Card('Hearts', '10')]

        self.assertTrue(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_one_card_invalid(self):
        last_trick = [Card('Hearts', '9')]
        new_trick = [Card('Hearts', '8')]
        cards_available = [Card('Hearts', '8')]

        self.assertFalse(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_one_card_different_suite(self):
        last_trick = [Card('Hearts', '9')]
        new_trick = [Card('Diamonds', '10')]
        cards_available = [Card('Diamonds', '10')]

        self.assertTrue(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_two_cards(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9')]
        new_trick = [Card('Diamonds', '10'), Card('Hearts', '10')]
        cards_available = [Card('Diamonds', '10'), Card('Hearts', '10')]

        self.assertTrue(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_two_cards_invalid_lower(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '8')]
        cards_available = [Card('Diamonds', '8'), Card('Hearts', '8')]

        self.assertFalse(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_two_cards_invalid_mismatch(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '9')]
        cards_available = [Card('Diamonds', '8'), Card('Hearts', '9')]

        self.assertFalse(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_three_cards(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9'), Card('Clubs', '9')]
        new_trick = [Card('Diamonds', '10'), Card('Hearts', '10'), Card('Clubs', '10')]
        cards_available = [Card('Diamonds', '10'), Card('Hearts', '10'), Card('Clubs', '10')]

        self.assertTrue(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_three_cards_invalid_lower(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9'), Card('Clubs', '9')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8')]
        cards_available = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8')]

        self.assertFalse(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_three_cards_invalid_mismatch(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9'), Card('Clubs', '9')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '9'), Card('Clubs', '9')]
        cards_available = [Card('Diamonds', '8'), Card('Hearts', '9'), Card('Clubs', '9')]

        self.assertFalse(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_simple_two_cards_bomb(self):
        last_trick = [Card('Hearts', '9'), Card('Diamonds', '9')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8'), Card('Spades', '8')]
        cards_available = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8'), Card('Spades', '8')]

        self.assertTrue(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_is_valid_trick_bomb_over_bomb(self):
        last_trick = [Card('Diamonds', '7'), Card('Hearts', '7'), Card('Clubs', '7'), Card('Spades', '7')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8'), Card('Spades', '8')]

        cards_available = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8'), Card('Spades', '8')]

        self.assertTrue(Game.is_valid_trick(last_trick, new_trick, cards_available))

    def test_comparing_straight_and_full_house(self):
        last_trick = [Card('Diamonds', '5'), Card('Hearts', '6'), Card('Clubs', '7'), Card('Spades', '8'), Card('Spades', '9')]
        new_trick = [Card('Diamonds', '8'), Card('Hearts', '8'), Card('Clubs', '8'), Card('Spades', 'A'), Card('Hearts', 'A')]

        self.assertTrue(GameRules.compare_tricks(last_trick, new_trick,[]))

if __name__ == "__main__":
    unittest.main()