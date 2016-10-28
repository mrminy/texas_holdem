"""
AI coded by Marius Amundsen
"""
import random

from marius.pokerklient import rules, deck
from player import Player


class ai(Player):
    """
    Used for testing og debugging. Always calls
    """

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board, round_nr):

        if (self.hand[0][1] + self.hand[1][1]) > 22 or (self.hand[0][1] == self.hand[1][1]):
            return max_bet

        getcardright = deck()
        card_id = getcardright.find_card_id(self.hand + board)
        til = rules(card_id, 1)
        score = til.see_winner()
        if score[1] > 0:
            if score[1] / 8 > current_bet:
                return score[1] / 25
            else:
                return max_bet

        if current_bet * 1.2 > pot:
            return -1
        else:
            return current_bet - self.bet


for a in range(10):
    my_randoms = random.sample(range(52), 2)
    # print(my_randoms)
    til = rules(my_randoms, 1)
    # print(til.see_winner())
