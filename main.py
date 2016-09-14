import game
import numpy as np
from player import Player, Call_player, Human_player
from ai import My_AI, My_Experimenter_AI, My_Experimenter_AI2

N_games = 100
win_counter = np.zeros(3)
total_calls, total_folds = 0, 0
for n in range(N_games):
    print(n)
    g = game.Texas_holdem()
    input_players = [My_Experimenter_AI2(0, "My AI", 0), Call_player(1, "Call player 1", 0), Call_player(2, "Call player 2", 0)]
    g.reset(input_players)
    while len(g.players) > 1:
        g.play_one_step()
    p_id = g.players[0].id_value
    win_counter[p_id] += 1
    total_calls += input_players[0].self_calls
    total_folds += input_players[0].self_folds

print(total_calls, total_folds)
print(win_counter)
