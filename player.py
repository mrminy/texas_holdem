import random

import parameters


class Player:
    def __init__(self, id_value, name, chips):
        """
        :param id_value: An int id of the player
        :param name:  The name of the player
        :param chips: Amount of chips to start with
        """
        self.id_value = id_value
        self.name = name
        self.hand = []  # two cards on hand. An example: [[0,2], [3,14]]
        self.blind = 0  # 0 = no blind, 1 = small blind, 2 = big blind
        self.chips = chips  # current amount of chips to bet
        self.bet = 0  # Current bet in this betting round
        self.total_bet = 0  # Current bet in this betting round

    def give_hand(self, card1, card2):
        self.hand = [card1, card2]

    def get_hand(self):
        return self.hand

    def make_decision(self, betting_history, current_bet, max_bet, players_left_this_round, players, pot, board):
        """
        This represents a total random player
        :param betting_history: betting history for this betting round
        :param current_bet: the current bet of the round
        :param max_bet: the maximum bet you can bet, based on the player with smallest amount of chips
        :param board: array of cards visible on the board
        :return: the bet. If fold, bet = -1. If check, bet = 0.
        """
        bet = -1
        action = random.randint(0, 1)
        if action == 1:
            call = random.random() >= 0.5
            if call:
                bet = current_bet - self.bet
            else:
                bet = random.randint(20, 200)
        return min(bet, self.chips)

    def __str__(self):
        return str(self.id_value) + ' - card 1: ' + str(self.hand[0]) + ' - card 2: ' + str(
            self.hand[1]) + ' - chips: ' + str(self.chips)


class Other_player(Player):
    """
    Used for testing og debugging
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_left_this_round, players, pot, board):
        if len(board) == 0:
            return current_bet - self.bet
        else:
            return 10


class Call_player(Player):
    """
    Used for testing og debugging. Always calls
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_left_this_round, players, pot, board):
        if current_bet == 0:
            return parameters.BIG_BLIND
        return current_bet - self.bet


class Human_player(Player):
    """
    Use this play for yourself against the other AIs. f = fold, c = check, b = bet
    (when bet is chosen, you can choose the amount of chips to bet)
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_left_this_round, players, pot, board):
        while True:
            action = input('Choose your action: (f=fold, c=check, b=bet) (current bet is ' + str(current_bet) + ')\n')

            if action == 'f':
                return -1
            elif action == 'c':
                if current_bet == 0 or current_bet == self.bet:
                    return 0
                else:
                    print("Can't select that action\n")
            else:
                try:
                    bet = int(input('Place your bet: (current bet is ' + str(current_bet) + ')\n'))
                    return bet
                except ValueError:
                    print("Not a number...")
