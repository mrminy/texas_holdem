import time

import game2 as game
import numpy as np
from player import Player, Call_player, Human_player, Other_player
from mikkel_ai import My_Experimenter_AI2, My_All_Inner
from marius_ai.marius_ai import ai as marius_ai
from johannes_ai import pokerAI as johannes_ai

start_time = time.time()
N_games = 100
win_counter = np.zeros(3)

for n in range(N_games):
    print("Game nr:", n)
    g = game.Texas_holdem(logger=False)
    input_players = [johannes_ai(0, "Johannes AI", 0), marius_ai(1, "Marius AI", 0), My_Experimenter_AI2(2, "Mikkel AI", 0)]
    g.reset(input_players)
    while len(g.players) > 1:
        g.play_one_step()
    p_id = g.players[0].id_value
    win_counter[p_id] += 1

print(win_counter)
print(time.time()-start_time)
