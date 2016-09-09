import game
from player import Player, Call_player
from ai import My_AI

N_games = 20
win_counter = 0

for n in range(N_games):
    print(n)
    g = game.Texas_holdem()
    input_players = [My_AI(0, "My AI", 0, None, None), Call_player(1, "Random player", 0, None, None)]
    g.reset(input_players)
    while len(g.players) > 1:
        g.play_one_step()
    if g.players[0].id_value == 0:
        win_counter += 1
print(win_counter)
