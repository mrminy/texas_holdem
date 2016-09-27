import time

import game
import numpy as np
from player import Player, Call_player, Human_player, Other_player
from mikkel_ai import My_Experimenter_AI2 as mikkel_ai
from marius_ai.marius_ai import ai as marius_ai
from johannes_ai import pokerAI as johannes_ai
# from mikkel_ai2 import My_Keras_SL_AI as mikkel_ai2

start_time = time.time()
N_games = 1
win_counter = np.zeros(4)

input_players = [johannes_ai("J"), marius_ai("Ma AI")]

for n in range(N_games):
    print("Game nr:", n, "- Current win rates:", win_counter)
    g = game.Texas_holdem(input_players=input_players, logger=False)
    while len(g.players) > 1:
        g.play_one_step()
    p_id = g.players[0].id_value
    win_counter[p_id] += 1

print(win_counter)
print(input_players)
print(time.time() - start_time)
