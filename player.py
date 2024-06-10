from typing import List, Union
from cards import Card, DogCard, DragonCard, PhoenixCard, MahJongCard


class Player:
    """
    Represents a player in the Tishu game.

    Attributes:
        name (str): The name of the player.
        hand (List[Union[Card, DogCard, DragonCard, PhoenixCard, OneCard]]): The player's current hand of cards.
    """

    def __init__(self, name: str):
        self.name = name
        self.hand = []

    def receive_cards(self, cards: List[Card]):
        """
        Adds received cards to the player's hand.

        Args:
            cards (List[Union[Card, DogCard, DragonCard, PhoenixCard, OneCard]]): A list of cards to add to the hand.
        """
        self.hand.extend(cards)

    def play_card(self, card: Card) -> List[Card]:
        if card not in self.hand:
            raise ValueError(f"Card {card} is not in hand")
        self.hand.remove(card)
        return card

    def __repr__(self):
        """
        Returns a string representation of the player.

        Returns:
            str: A string representing the player.
        """
        return f"Player(name={self.name}, hand={self.hand})"


class AIPlayer(Player):
    """
    Represents an AI player in the Tishu game, inheriting from Player.

    Methods:
        make_move(): Determines and returns the AI's move.
    """

    def __init__(self, name: str):
        super().__init__(name)

    def make_move(self) -> Union[Card, DogCard, DragonCard, PhoenixCard, MahJongCard]:
        """
        Determines the AI's move. This is a simple implementation and can be extended with more complex logic.

        Returns:
            Union[Card, DogCard, DragonCard, PhoenixCard, OneCard]: The card chosen to play.
        """
        # For simplicity, the AI will play the first card in its hand.
        # More sophisticated strategies can be implemented here.
        if not self.hand:
            raise ValueError("AI has no cards to play")
        return self.play_card(self.hand[0])


if __name__ == "__main__":
    # Testing the Player and AIPlayer classes
    from cards import Deck

    # Create players
    player1 = Player("Alice")
    ai_player = AIPlayer("AI")

    # Create and shuffle deck
    deck = Deck()

    # Deal hands to players
    player1.receive_cards(deck.deal_hand(5))
    ai_player.receive_cards(deck.deal_hand(5))

    print(player1)
    print(ai_player)

    # Players play cards
    try:
        print(f"{player1.name} plays: {player1.play_card(player1.hand[0])}")
        print(f"{ai_player.name} plays: {ai_player.make_move()}")
    except ValueError as e:
        print(e)

    print(player1)
    print(ai_player)