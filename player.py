import random

import parameters


class Player:
    def __init__(self, name):
        """
        :param id_value: An int id of the player
        :param name:  The name of the player
        :param chips: Amount of chips to start with
        """
        self.id_value = -1
        self.name = name
        self.hand = []  # two cards on hand. An example: [[0,2], [3,14]]
        self.blind = 0  # 0 = no blind, 1 = small blind, 2 = big blind
        self.chips = 0  # current amount of chips to bet
        self.bet = 0  # Current bet in this betting round

    def give_hand(self, card1, card2):
        self.hand = [card1, card2]

    def reset(self):
        self.hand = []
        self.blind = 0
        self.chips = 0
        self.bet = 0

    def get_hand(self):
        return self.hand

    def get_open_information(self):
        return {'id_value': self.id_value, 'name': self.name, 'blind': self.blind, 'chips': self.chips, 'bet': self.bet}

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board):
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
        return str(self.name) + ' - ' + str(self.id_value) + ' - card 1: ' + str(self.hand[0]) + ' - card 2: ' +\
               str(self.hand[1]) + ' - blind: ' + str(self.blind) + ' - chips: ' + str(self.chips) + ' - current bet: '\
               + str(self.bet)


class Other_player(Player):
    """
    Used for testing og debugging
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board):
        if len(board) == 0:
            return current_bet - self.bet
        else:
            return 10


class Call_player(Player):
    """
    Used for testing og debugging. Always calls
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board):
        if current_bet == 0:
            return parameters.BIG_BLIND
        return current_bet - self.bet


class Raiser_player(Player):
    """
    Used for testing og debugging. Always calls
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board):
        bet = min(parameters.BIG_BLIND, self.chips)
        return bet * 2


class Human_player(Player):
    """
    Use this play for yourself against the other AIs. f = fold, c = check, b = bet
    (when bet is chosen, you can choose the amount of chips to bet)
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board):
        while True:
            action = input('Choose your action: (f=fold, c=check, b=bet) (current bet is ' + str(current_bet) + ')\n')

            if action == 'f':
                return -1
            elif action == 'c':
                if current_bet == 0 or current_bet == self.bet:
                    return 0
                else:
                    print("Can't select that action\n")
            elif action == 'b':
                try:
                    bet = 0
                    while bet < current_bet:
                        bet = int(input('Place your bet: (current bet is ' + str(current_bet) + ')\n'))
                        if bet < current_bet:
                            print("Can't bet less than current bet. Current bet is", current_bet)
                    return bet
                except ValueError:
                    print("Not a number...")
