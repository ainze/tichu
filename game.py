
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

    @classmethod
    def is_valid_trick(cls, last_trick: List[Card], new_trick: List[Card], cards_available: List[Card]) -> bool:
        # Check if the player has the cards they want to play
        for card in new_trick:
            if card not in cards_available:
                return False

        if len(last_trick) == 0:
            return True # any card is valid if this is the first hand to play
        if Game.is_bomb(new_trick):
            if not Game.is_bomb(last_trick):
                return True #bombs can always be played
            else:
                return Game.compare_bombs(new_trick, last_trick)

        # If the last trick is a bomb and the new trick is not, the new trick is invalid
        if Game.is_bomb(last_trick) and not Game.is_bomb(new_trick):
            return False

        # Check if the new trick matches the type and is higher in value than the last trick
        return Game.compare_tricks(new_trick, last_trick)

    @classmethod
    def is_bomb(cls, new_trick: List[Card]) -> bool:
        if len(new_trick) == 4:
            # check if all 4 are same rank
            if any(card.rank != new_trick[0].rank for card in new_trick):
                return False # this is not a bomb since rank is different
            else:
                return True
        elif len(new_trick) > 4:
            if any(card.suit != new_trick[0].suit for card in new_trick):
                return False # cards need to be same suite for straight

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

