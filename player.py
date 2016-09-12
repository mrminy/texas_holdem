import random


class Player:
    def __init__(self, id_value, name, chips, card1, card2):
        self.id_value = id_value
        self.name = name
        self.hand = []
        self.blind = 0
        self.chips = chips
        self.bet = 0
        self.give_hand(card1, card2)

    def give_hand(self, card1, card2):
        self.hand = [card1, card2]

    def get_hand(self):
        return self.hand

    def make_decision(self, betting_history, current_bet, max_bet, pot, board, allowed_actions):
        """
        This represents a total random player
        :param betting_history: betting history for this betting round
        :param current_bet: the current bet of the round
        :param max_bet: the maximum bet you can bet, based on the player with smallest amount of chips
        :param board: array of cards visible on the board
        :param allowed_actions: The number of allowed actions. 1 = bet, 2 = fold, 3 = check
        :return: the bet. If fold, bet = -1. If check, bet = 0.
        """
        bet = -1
        action = random.randint(1, allowed_actions)
        if action == 1:
            call = random.random() >= 0.5
            if call:
                bet = current_bet - self.bet
            else:
                bet = random.randint(20, 200)
        elif action == 3 or current_bet == 0:
            bet = 0
        return min(bet, self.chips)

    def __str__(self):
        return str(self.id_value) + ' - card 1: ' + str(self.hand[0]) + ' - card 2: ' + str(
            self.hand[1]) + ' - chips: ' + str(self.chips)


class Other_player(Player):
    """
    Used for testing og debugging
    """

    def make_decision(self, betting_history, current_bet, max_bet, pot, board, allowed_actions):
        if len(board) == 0:
            return current_bet - self.bet
        else:
            return 10


class Call_player(Player):
    """
    Used for testing og debugging. Always calls
    """

    def make_decision(self, betting_history, current_bet, max_bet, pot, board, allowed_actions):
        return current_bet - self.bet
