import game
import numpy as np
from player import Player, Call_player
from ai import My_AI, My_Experimenter_AI, My_Experimenter_AI2

N_games = 20
win_counter = np.zeros(2)

for n in range(N_games):
    print(n)
    g = game.Texas_holdem()
    input_players = [My_Experimenter_AI2(0, "My AI", 0, None, None), Call_player(1, "My experimenter AI", 0, None, None)]
    g.reset(input_players)
    while len(g.players) > 1:
        g.play_one_step()
    p_id = g.players[0].id_value
    win_counter[p_id] += 1

print(input_players[0].self_calls, input_players[0].self_folds)
print(win_counter)
