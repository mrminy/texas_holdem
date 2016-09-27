"""
AI written by Mikkel Nylend
"""

import time

import deck
import evaluator
import parameters
from player import Player


# 08.09. 20:00 --> 9562 evaluations per second (tested with pocket ace)
# 11.09. 20:00 --> 10000 evaluations per second (tested with pocket ace)
# 12.09. 12:00 --> 30000-40000 evaluations per second (tested with pocket ace & 4 threads)
# 12.09. 21:00 --> Raw test with 7 new random cards picked each time reveals about 19k evaluations per second


def evaluate_situation(N, n_opponents, my_hand, board):
    start_time = time.time()

    wins, losses, draws = 0.0, 0.0, 0.0
    board_len = len(board)

    # Test win-ratio with cards
    for i in range(N):
        d = deck.Deck()
        my_card_1 = d.draw_specific_card(my_hand[0][0], my_hand[0][1])
        my_card_2 = d.draw_specific_card(my_hand[1][0], my_hand[1][1])

        if board_len >= 1:
            board_1 = d.draw_specific_card(board[0][0], board[0][1])
        else:
            board_1 = d.draw_card()
        if board_len >= 2:
            board_2 = d.draw_specific_card(board[1][0], board[1][1])
        else:
            board_2 = d.draw_card()
        if board_len >= 3:
            board_3 = d.draw_specific_card(board[2][0], board[2][1])
        else:
            board_3 = d.draw_card()
        if board_len >= 4:
            board_4 = d.draw_specific_card(board[3][0], board[3][1])
        else:
            board_4 = d.draw_card()
        if board_len == 5:
            board_5 = d.draw_specific_card(board[4][0], board[4][1])
        else:
            board_5 = d.draw_card()

        opponents_card_1 = d.draw_n_cards(n_opponents)
        opponents_card_2 = d.draw_n_cards(n_opponents)

        my_cards = [my_card_1, my_card_2, board_1, board_2, board_3, board_4, board_5]
        opponent_cards = []
        for i in range(n_opponents):
            opponent_cards.append(
                [opponents_card_1[i], opponents_card_2[i], board_1, board_2, board_3, board_4, board_5])
        my_score = evaluator.evaluate(my_cards)
        player_score_map = [[my_score, 0]]
        counter = 1
        for opponent_c in opponent_cards:
            player_score_map.append([evaluator.evaluate(opponent_c), counter])
            counter += 1
        player_score_map.sort(key=lambda x: x[0], reverse=True)
        top_score = player_score_map[0][0]
        top_players = [player_score_map[0][1]]
        counter = 1
        while counter < n_opponents + 1 and player_score_map[counter][0] == top_score:
            top_players.append(player_score_map[counter][1])
            counter += 1
        if 0 in top_players and len(top_players) == 1:
            wins += 1.0
        elif 0 in top_players:
            draws += 1.0
        else:
            losses += 1.0

    end_time = time.time()
    time_used = end_time - start_time

    # print("Runs:", str(N), "Wins:", str(wins), "Losses:", str(losses), "Draws:", str(draws))
    # print("Win %:", wins / N, "Loose %:", losses / N, "Draw %:", draws / N, my_hand)
    # print("Time used:", str(time_used))
    # print("Evaluations per second:", N / time_used)
    return wins / N, losses / N, draws / N


class My_Experimenter_AI2(Player):
    def __init__(self, name):
        super().__init__(name)
        self.self_calls = 0
        self.self_folds = 0

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board, round_nr):
        this_bet = 0
        pot_odds = 0

        win_ratio, loose_ratio, tie_ratio = evaluate_situation(parameters.EVALUATION_PRECISION,
                                                               len(players_this_round) - 1, self.hand, board)
        win_ratio += tie_ratio
        if win_ratio != 0:
            equity = loose_ratio / win_ratio
        else:
            equity = float('inf')
        round_score = [0.7, 0.8, 0.8, 0.8, 0.9, 0.95]
        equity *= round_score[len(board)]

        if current_bet - max(0, self.bet) != 0:
            what_i_have_to_bet = current_bet - self.bet
            what_im_been_offered = pot
            pot_odds = what_im_been_offered / what_i_have_to_bet

            # print("Pot - equity:", pot_odds, pot, current_bet, self.bet, equity)
            if pot_odds > equity:
                if win_ratio >= 0.85:
                    this_bet = self.chips
                else:
                    this_bet = current_bet - self.bet
        else:
            this_bet = min(0.2 * parameters.START_CHIPS, self.chips)
        if this_bet > 0:
            self.self_calls += 1
        else:
            self.self_folds += 1
        if parameters.RECORD_GAMES:
            game_recorder(win_ratio, loose_ratio, tie_ratio, pot_odds, current_bet, self.bet, self.chips, this_bet)
        return this_bet


def game_recorder(win_rate, loose_rate, tie_rate, pot_odds, current_bet, player_bet, player_chips, selected_bet):
    # fold, check, call, raise
    if selected_bet == 0:
        action = [0, 1, 0, 0, 0]
    elif selected_bet + player_bet == current_bet:
        action = [0, 0, 1, 0, 0]
    elif selected_bet == player_chips:
        action = [0, 0, 0, 0, 1]
    elif selected_bet + player_bet > current_bet:
        action = [0, 0, 0, 1, 0]
    else:
        action = [1, 0, 0, 0, 0]

    state = [win_rate, loose_rate, tie_rate, pot_odds]
    with open("inputs.txt", "a") as inputfile:
        inputfile.write(str(state) + "\n")
    with open("answers.txt", "a") as answerfile:
        answerfile.write(str(action) + "\n")


if __name__ == '__main__':
    print(evaluate_situation(parameters.EVALUATION_PRECISION, 1, [[0, 14], [1, 14]], []))
