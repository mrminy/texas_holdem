import game
import numpy as np
from player import Player, Call_player, Human_player, Other_player
from ai import My_Experimenter_AI2, My_All_Inner

N_games = 100
win_counter = np.zeros(3)
total_calls, total_folds = 0, 0
for n in range(N_games):
    print(n)
    g = game.Texas_holdem(logger=False)
    input_players = [My_Experimenter_AI2(0, "My experimenter AI2", 0), Call_player(1, "Call player 1", 0)]
    g.reset(input_players)
    while len(g.players) > 1:
        g.play_one_step()
    p_id = g.players[0].id_value
    win_counter[p_id] += 1

print(total_calls, total_folds)
print(win_counter)
