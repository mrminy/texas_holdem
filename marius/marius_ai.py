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


        def find_Winner_chance(precison, players, cardsin):
            times_won= 0
            my_hand= [cardsin[0],cardsin[1]]
            board= [0,0,0,0,0]
            a_board = []


            for a in range(precison):
                r_cards =random.sample(range(51),40)
                for a in cardsin:
                    try:
                        r_cards.remove(a)
                    except ValueError:
                        pass
                for a in range(4):
                    try:
                        board[a] =  cardsin[a+2]
                    except:
                        board[a] = r_cards[a]
                a_board =[my_hand[0],my_hand[1],board[0],board[1],board[2],board[3],board[4]]
                for a in range(players):
                    a_board.extend([r_cards[a*7+7],r_cards[a*7+8],r_cards[a*7+9],r_cards[a*7+10],r_cards[a*7+11],r_cards[a*7+12],r_cards[a*7+13]])

                analyse = rules(a_board, players)
                score = analyse.see_winner()




                if score[0]< 1:
                    times_won +=1
            probability= times_won/precison
            return probability

        getcardright = deck()
        card_id = getcardright.find_card_id(self.hand + board)
        til = rules(card_id, 1)
        score = til.see_winner()
        hei= find_Winner_chance(1000,len(players_this_round),card_id)


        bet_now = current_bet- self.bet
        my_value_economy= bet_now/pot

        my_value_Random = hei * len(players_this_round)



        if current_bet- self.bet> 500 :
            if my_value_Random>1.5:
                return max_bet
            elif my_value_Random>1.32:
                return current_bet *4.5
            elif my_value_Random>1.31:
                return current_bet - self.bet
            elif my_value_Random>1:
                return 0
            elif my_value_Random>0.87:
                return 0





        if 500>current_bet- self.bet> 0 :
            if my_value_Random>1.56:
                    return max_bet
            elif my_value_Random>1.32:
                return current_bet - self.bet
            elif my_value_Random>1.21:
                return current_bet - self.bet
            elif my_value_Random>1.11:
                return current_bet - self.bet
            elif my_value_Random>0.87:
                return 0




        if my_value_Random>1.5:
            return max_bet
        elif my_value_Random>1.22:
            return current_bet *4.5
        elif my_value_Random>1.11:
            return current_bet *2.5
        elif my_value_Random>1:
            return current_bet *1.5
        elif my_value_Random>0.87:
            return current_bet - self.bet
        else:
            if bet_now < 10:
                return bet_now
            else:return 0


        if (self.hand[0][1] + self.hand[1][1]) > 22 or (self.hand[0][1] == self.hand[1][1]):
            return max_bet


        if score[1] > 0:
            if score[1] / 8 > current_bet:
                return score[1] / 25
            else:
                return max_bet

        if current_bet * 1.2 > pot:
            return -1
        else:
            return current_bet - self.bet

