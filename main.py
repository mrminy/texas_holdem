import time

import game
import numpy as np
from player import Player, Call_player, Human_player, Other_player
from mikkel_ai import My_Experimenter_AI2 as mikkel_ai
from marius_ai.marius_ai import ai as marius_ai
from johannes_ai import pokerAI as johannes_ai

start_time = time.time()
N_games = 1000
win_counter = np.zeros(3)

input_players = []

for n in range(N_games):
    print("Game nr:", n, "- Current win rates:", win_counter)
    g = game.Texas_holdem(logger=False)
    input_players = [mikkel_ai(0, "Mikkel AI", 0), johannes_ai(1, "Johannes AI", 0), marius_ai(2, "Marius AI", 0)]
    g.reset(input_players)
    while len(g.players) > 1:
        g.play_one_step()
    p_id = g.players[0].id_value
    win_counter[p_id] += 1

print(win_counter)
print(input_players)
print(time.time()-start_time)
