import logging
import unittest
from game import Game
from cards import Card
from player import RandomPlayer


class TestFullGame(unittest.TestCase):
    def setUp(self):
        self.game = Game([RandomPlayer("P1"), RandomPlayer("P2"), RandomPlayer("P3"), RandomPlayer("P4")])

    def test_is_bomb_same_rank(self):
        self.game.deal()

        self.game.round_loop()

        logging.info(self.game)

        total_cards = []
        for i in range(4):
            total_cards.extend(self.game.players[i].stored_cards)

        logging.info(f'Total cards={len(total_cards)}')

if __name__ == "__main__":
    unittest.main()