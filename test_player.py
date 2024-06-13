import logging
import unittest

from cards import Card
from player import RandomPlayer


class PlayerCase(unittest.TestCase):
    def test_valid_plays(self):
        player = RandomPlayer("testAI")

        player.rec_cards([Card('Diamonds', '9'), Card('Clubs', '9'), Card('Hearts', '9'), Card('Spades', '9')])

        possible_plays = player.generate_possible_plays()
        logging.info('possible plays: {}'.format(possible_plays))
        self.assertTrue(any(possible_play == [Card('Diamonds', '9')] for possible_play in possible_plays))
        self.assertTrue(any(possible_play == [Card('Diamonds', '9'), Card('Clubs', '9')] for possible_play in possible_plays))
        self.assertTrue(any(possible_play == [Card('Diamonds', '9'), Card('Clubs', '9'), Card('Hearts', '9')] for possible_play in possible_plays))
        self.assertTrue(any(possible_play == [Card('Diamonds', '9'), Card('Clubs', '9'),
                                              Card('Hearts', '9'), Card('Spades', '9')]
                            for possible_play in possible_plays))




    def test_valid_plays_mahjong(self):
        player = RandomPlayer("testAI")

        player.rec_cards([Card('Special', 'MahJong'), Card('Clubs', '9'), Card('Hearts', '9'), Card('Spades', '9')])

        possible_plays = player.generate_possible_plays()
        logging.info('possible plays: {}'.format(possible_plays))
        self.assertTrue(any(possible_play == [Card('Clubs', '9')] for possible_play in possible_plays))
        self.assertTrue(
            any(possible_play == [Card('Clubs', '9'), Card('Spades', '9')] for possible_play in possible_plays))
        self.assertTrue(any(
            possible_play == [Card('Clubs', '9'), Card('Hearts', '9'), Card('Spades', '9')] for possible_play in
            possible_plays))
        self.assertFalse(any(possible_play == [Card('Diamonds', '9'), Card('Clubs', '9'),
                                              Card('Hearts', '9'), Card('Spades', '9')]
                            for possible_play in possible_plays))
        self.assertTrue(any(possible_play == [Card('Special', 'MahJong')] for possible_play in possible_plays))

        logging.info('possible plays: {}'.format(possible_plays))

if __name__ == '__main__':
    unittest.main()
