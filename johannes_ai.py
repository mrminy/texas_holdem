"""
AI coded by Johannes Barstad
"""

import evaluator
from player import Player


class pokerAI(Player):
    """
    Used for testing og debugging. Always calls
    """

    def __init__(self, name):
        super().__init__(name)
        self.global_confidence = 0
        self.previous_flop = -1
        self.previous_flop_counter = 0

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board):
        if len(board) == 0 and self.previous_flop_counter == 0:
            self.global_confidence = 0
        if self.previous_flop == len(board):
            self.previous_flop_counter += 1
        else:
            self.previous_flop_counter = 0
        self.previous_flop = len(board)

        confidence = 0

        # confidence += ((self.chips-1000)**1.5)**0.5

        # Tallverdier på kort
        kort_verdi1 = self.hand[0][1]
        kort_verdi2 = self.hand[1][1]
        # suit på kort
        kort_suit1 = self.hand[0][0]
        kort_suit2 = self.hand[1][0]

        big_blind_confidence = 50
        small_blind_confidence = 25
        hoye_kort_confidence = 50
        hoyt_kort_confidence = 20
        lave_kort_confidence = -200
        lavt_kort_confidence = -50
        par_confidence = 250
        suit_confidence = 50
        ett_ess_confidence = 150

        # Big blind
        if self.blind == 2:
            confidence += big_blind_confidence
        if self.blind == 1:
            confidence += small_blind_confidence

        # Høye kort
        hoye_kort = kort_verdi1 > 10 and kort_verdi2 > 10

        if hoye_kort:
            confidence += hoye_kort_confidence

        # Et høyt kort
        hoyt_kort = kort_verdi1 > 10 or kort_verdi2 > 10

        if hoyt_kort:
            confidence += hoyt_kort_confidence

        # Lave kort
        lave_kort = kort_verdi1 <= 10 and kort_verdi2 <= 10

        if lave_kort:
            confidence += lave_kort_confidence

        # Lavt kort
        lavt_kort = kort_verdi1 <= 10 or kort_verdi2 <= 10

        if lavt_kort:
            confidence += lavt_kort_confidence

        # Lik suit
        liksuit = kort_suit1 == kort_suit2

        if liksuit:
            confidence += suit_confidence

        # Ett ess
        ett_ess = kort_verdi1 == 14 or kort_verdi2 == 14

        if ett_ess:
            confidence += ett_ess_confidence

        par = kort_verdi1 == kort_verdi2

        if par:
            confidence += par_confidence

        # Høyt par
        if par and hoye_kort:
            confidence += hoye_kort_confidence * 3

        # to billedkort lik suit
        if hoye_kort and liksuit:
            confidence += suit_confidence * 2

        # lik suit, ett es
        if ett_ess and liksuit:
            confidence += suit_confidence * 3

        # Evaluere hånd og betting
        cards = self.hand + board

        fold = -1
        low_conf_bet = 0.2 * self.chips
        high_conf_bet = 0.3 * self.chips
        good_river_bet = self.chips * 0.5

        self.global_confidence += confidence

        # print("CONFIDENCE:", confidence, "GLOBAL_CONFIDENCE:", self.global_confidence)
        if len(cards) == 5:
            evaluate_cards = evaluator.simple_evaluate(cards)
            if evaluate_cards > 2:
                self.global_confidence += 200
                return max(good_river_bet, current_bet)
        if len(cards) > 5:
            evaluate_cards = evaluator.simple_evaluate(cards)
            if evaluate_cards > 3:
                self.global_confidence += 300
                return max(good_river_bet, current_bet)

        if self.global_confidence < 0:
            return fold
        elif 0 < self.global_confidence <= 100:
            return min(low_conf_bet, current_bet)
        elif 100 < self.global_confidence <= 200:
            return max(high_conf_bet, current_bet)
        elif self.global_confidence > 200:
            return self.chips
