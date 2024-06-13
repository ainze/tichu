import logging
from typing import List

from cards import Deck, Card
from player import Player, RandomPlayer


class Game:

    def __init__(self, players: List[Player]):
        self.players = players

        self.dealer = 0
        self.deck = Deck()
        self.current_player = 0

    def deal(self):
        self.deck.shuffle()
        logging.info(f'deck is {len(self.deck.cards)}')
        # give first 8 and ask for GrandTichu
        for i in range(4):
            self.players[(i + self.dealer) % 4].rec_cards(self.deck.deal_hand(8))
            self.players[(i + self.dealer) % 4].ask_grandTichu()

        # give next 6 - deal finished.
        for i in range(4):
            self.players[(i + self.dealer) % 4].rec_cards(self.deck.deal_hand(6))

        for i in range(4):
            logging.info(f'player has {len(self.players[i].cards)} cards')
            if self.players[i].hasMahJong():
                self.current_player = i

        logging.info(f'deck is {len(self.deck.cards)}')

    def round_loop(self):
        round_not_finished = True
        cards_on_table = [[]]
        players_passed = [False, False, False, False]
        players_no_more_cards = []
        players_tichu = [False, False, False, False]

        while round_not_finished:
            action = self.players[self.current_player].ask_trick(cards_on_table)
            if action.action == 'PASS':
                players_passed[self.current_player] = True

            else:
                # Did not pass, so played a card
                if action.action == 'TICHU':
                    players_tichu[self.current_player] = True

                cards_on_table.append(action.cards)

                if len(self.players[self.current_player].cards) == 0:
                    logging.info(
                        f'Player {self.players[self.current_player].name} played his last card so he is now flagged PASS')
                    players_passed[self.current_player] = True
                    players_no_more_cards.append(self.players[self.current_player])

                    if len(players_no_more_cards) == 3:
                        logging.info(f'Three players without cards')
                        round_not_finished = False
                    # check if team is done
                    elif (self.players[0].hasNoCards() and self.players[2].hasNoCards()) or (
                            self.players[1].hasNoCards() and self.players[3].hasNoCards()):
                        logging.info(f'Team has finished first')
                        round_not_finished = False

                logging.info(f'Player {self.players[self.current_player].name} has played {action.cards}')

            if players_passed.count(True) == 3:
                # All players passed, it is again to the current player.
                # but first some cleanup and storing of cards
                logging.debug(f'Three players passed')
                players_passed = [False, False, False, False]
                self.players[self.current_player].store_cards([item for sublist in cards_on_table for item in sublist])
                cards_on_table = [[]]
            else:
                # not all have passed so we continue
                self.current_player = (self.current_player + 1) % 4

            if not round_not_finished:
                for player in players_no_more_cards:
                    logging.debug(f'Player {player.name} no longer has cards')
                # take remaining cards and give it to first player
                for player in self.players:
                    logging.debug(f'Player {player.name} has {len(player.cards)} cards and {len(player.stored_cards)} stored cards')
                    if player not in players_no_more_cards:
                        logging.debug(f'player {player.name} his cards {player.cards} are given '
                                      f'to first player {players_no_more_cards[0].name}')
                        players_no_more_cards[0].store_cards(player.cards)
                        player.cards = []
                        logging.debug(
                            f'Player {player.name} now has {len(player.cards)} cards and {len(player.stored_cards)} stored cards')


    def __repr__(self):
        return f"GameState: {self.players}"

    def calculate_score(self):
        player_scores = [0, 0, 0, 0]
        team_scores = [0, 0]

        for i, player in enumerate(self.players):
            for card in player.stored_cards:
                if card == Card('Special', 'Dragon'):
                    player_scores[i] += 25
                elif card == Card('Special', 'Phoenix'):
                    player_scores[i] -= 25
                elif card == Card('Special', 'MahJong'):
                    continue
                elif card == Card('Special', 'Dog'):
                    continue  # Dog card does not score any points
                elif card.rank in ['5', '10', 'K']:
                    if card.rank == '5':
                        player_scores[i] += 5
                    elif card.rank == '10':
                        player_scores[i] += 10
                    elif card.rank == 'K':
                        player_scores[i] += 10

        team_scores[0] = player_scores[0] + player_scores[2]
        team_scores[1] = player_scores[1] + player_scores[3]

        # Additional scoring rules can be applied here if needed

        logging.info(team_scores)

    def finish_round(self):

        self.calculate_score()
        self.deck = Deck()
        pass


if __name__ == '__main__':
    game = Game([RandomPlayer("P1"), RandomPlayer("P2"), RandomPlayer("P3"), RandomPlayer("P4")])
    game.deal()
    print(f"Dealed! Currentplayer: {game.current_player}")
    for i in range(4):
        print(f"Player {game.players[i].name} has {len(game.players[i].cards)} cards")
    game.round_loop()
    print(f"Hand finished! Dealer is now {game.players[game.dealer].name}")
    #game.round_loop()
    #print(f"Hand finished! Dealer is now {game.players[game.dealer].name}")
