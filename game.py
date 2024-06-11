
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

        #give first 8 and ask for GrandTichu
        for i in range(4):
            self.players[(i + self.dealer)%4].rec_cards(self.deck.deal_hand(8))
            self.players[(i + self.dealer) % 4].ask_grandTichu()

        #give next 6 - deal finished.
        for i in range(4):
            self.players[(i + self.dealer)%4].rec_cards(self.deck.deal_hand(6))

        for i in range(4):
            if (self.players[i].hasMahJong()):
                self.current_player = i

    def round_loop(self):
        round_not_finished = True
        cards_on_table = [[]]
        players_passed = [False, False, False, False]
        players_no_more_cards = [False, False, False, False]
        players_tichu = [False, False, False, False]


        while(round_not_finished):
            action = self.players[self.current_player].ask_trick(cards_on_table)
            if action.action == 'PASS':
                players_passed[self.current_player] = True

            else:
                # Did not pass, so played a card
                if action.action == 'TICHU':
                    players_tichu[self.current_player] = True
                cards_on_table.append(action.cards)
                if len(self.players[self.current_player].cards) == 0:
                    print(f"Player {self.players[self.current_player].name} played his last card so he is now flagged PASS")
                    players_passed[self.current_player] = True
                    players_no_more_cards[self.current_player] = True
                print(f"Player {self.players[self.current_player].name} has played {action.cards}")

            if players_no_more_cards.count(True) == 3:
                # TODO: hier moeten we nog checken of een team klaar is
                round_not_finished = False
            elif players_passed.count(True) == 3:
                # All players passed, it is again to the current player.
                # but first some cleanup and storing of cards
                players_passed= [False, False, False, False]
                self.players[self.current_player].store_cards([item for sublist in cards_on_table for item in sublist])
                cards_on_table = [[]]
            else:
                #not all have passed so we continue
                self.current_player = (self.current_player + 1) % 4



    def __repr__(self):
        return f"GameState: {self.players}"

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

