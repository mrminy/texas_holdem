import random

import datetime
import numpy as np
import time

import game
import parameters
import matplotlib.pyplot as plt
import self_play_env

from scipy.interpolate import spline
from player import Call_player
from mikkel_ai import My_Experimenter_AI2 as mikkel_ai
from marius_ai.marius_ai import ai as marius_ai
from johannes_ai import pokerAI as johannes_ai
from mikkel_ai2 import My_Keras_SL_AI, My_Keras_SL_AI_Self_Play_Learner

"""
Implementation of deep Q-learning with self play (or play against others)
- Fetch current policy P
- For N games
    - Pick opponent O (randomly)
    - Create initialization state S
    - Play until end against O
        - Select action A (epsilon greedy)
        - Fetch reward R after doing A
        - Fetch new state S' after doing A
        - Improve P with RL algorithm based on reward
        - Current state = new state
"""
N_GAMES = 1000
MIN_EXPERIENCE_REPLAY_SIZE = 500
MAX_EXPERIENCE_REPLAY_SIZE = 10000
BATCH_SIZE = 100
NUM_STATES = 4
NUM_ACTIONS = 5
EPSILON_DEGRADE = parameters.EPSILON / N_GAMES
GAMMA = 0.99
LOGGER = False
TRAIN = True
SAVE_NEW_MODEL = True
PLOT_REWARD_GRAPH = True
PLOT_TOTAL_REWARD_GRAPH = True

def main():
    for i in range(N_GAMES):
        agent = My_Keras_SL_AI_Self_Play_Learner("New Keras AI")
        opponent = fetch_opponent()
        spe = self_play_env.self_play_env(agent, opponent)


def fetch_opponent():
    """
    :return: a randomly selected opponent
    """
    possible_opponents = [Call_player("Call player"), mikkel_ai("Mikkel AI"), johannes_ai("Johannes AI"),
                          marius_ai("Marius AI"),
                          My_Keras_SL_AI("Old Keras AI", model_path='data/old_models/my_model_relu_20_20_dropout.h5')]
    return random.choice(possible_opponents)


def improve_agent(agent, replay):
    len_mini_batch = min(len(replay), BATCH_SIZE)
    mini_batch = random.sample(replay, len_mini_batch)
    X_train = np.zeros((len_mini_batch, NUM_STATES))
    Y_train = np.zeros((len_mini_batch, NUM_ACTIONS))
    for index_rep in range(len_mini_batch):
        new_rep_state, reward_rep, action_rep, done_rep, old_rep_state = mini_batch[index_rep]
        old_rep_state = np.array(old_rep_state)
        new_rep_state = np.array(new_rep_state)
        old_q = agent.model.predict(old_rep_state.reshape(1, NUM_STATES))[0]
        new_q = agent.model.predict(new_rep_state.reshape(1, NUM_STATES))[0]
        if LOGGER:
            print("After predict", old_q, new_q)
        update_target = np.copy(old_q)
        if done_rep:
            update_target[action_rep] = -1
        else:
            update_target[action_rep] = reward_rep + (GAMMA * np.max(new_q))
        X_train[index_rep] = old_rep_state
        Y_train[index_rep] = update_target
        loss = agent.model.train_on_batch(X_train, Y_train)
        return loss


def main2():
    start_time = time.time()
    win_counter = np.zeros(2)
    training_step = 0
    replay = []
    total_reward_history = []
    reward_history = []
    agent = My_Keras_SL_AI_Self_Play_Learner("New Keras AI")
    for n in range(N_GAMES):
        reward_game = 0
        opponent = fetch_opponent()
        g = game.Texas_holdem(input_players=[agent, opponent], logger=False)
        current_state = agent.get_state(g.get_state())
        chips_before = 0
        total_bet = 0
        zero_deal_nr_count = 0

        while len(g.players) > 1:
            if len(g.players_this_round) == 1 and g.next_players_turn() == agent.id_value:
                reward = (agent.chips - chips_before) / chips_before
                reward_history.append(reward)
                reward_game += reward
                # Training batch
                if TRAIN and len(replay) > MIN_EXPERIENCE_REPLAY_SIZE:
                    training_step += 1
                    loss_train = improve_agent(agent, replay)[0]
                    if LOGGER:
                        print("Training batch:", training_step, "- Reward:", reward, "- Total reward:", reward_game,
                              "- Loss:", loss_train, "- Replay size:", len(replay))

            if g.deal_nr == 0 and zero_deal_nr_count == 0:
                chips_before = agent.chips + agent.bet
                total_bet = agent.bet
                zero_deal_nr_count += 1
            elif g.deal_nr != 0:
                zero_deal_nr_count = 0

            if g.next_players_turn() == agent.id_value:
                bet, action = agent.get_action(g.get_state())
                g.play_one_step(bet)
                if agent.bet == -1:
                    total_bet = 0
                else:
                    total_bet += agent.bet
                next_state = agent.get_state(g.get_state())
                if len(g.players_this_round) == 1:
                    total_bet = 0
                reward = (agent.chips + total_bet - chips_before) / chips_before
                reward_history.append(reward)
                reward_game += reward
                replay.append([next_state, reward, action, len(g.players) > 1, current_state])
                if LOGGER:
                    print("Round:", g.round_nr, "- Deal:", g.deal_nr, "- Reward:", reward, "- Action:", action,
                          "- Current state:", current_state)

                # Training batch
                if TRAIN and len(replay) > MIN_EXPERIENCE_REPLAY_SIZE:
                    training_step += 1
                    loss_train = improve_agent(agent, replay)[0]
                    if LOGGER:
                        print("Training batch:", training_step, "- Reward:", reward, "- Total reward:", reward_game,
                              "- Loss:", loss_train, "- Replay size:", len(replay))

                # Popping experiences if replay is too big
                if len(replay) > MAX_EXPERIENCE_REPLAY_SIZE:
                    replay.pop(np.random.randint(MAX_EXPERIENCE_REPLAY_SIZE) + 1)
                current_state = next_state
            else:
                g.play_one_step()
        total_reward_history.append(reward_game)

        if g.players[0].id_value == 0:
            win_counter[0] += 1
        else:
            win_counter[1] += 1

        if parameters.EPSILON > parameters.MIN_EPSILON:
            parameters.EPSILON -= EPSILON_DEGRADE

        print(datetime.datetime.now(), n, "-", win_counter, "-", len(replay), "-", parameters.EPSILON, "-", "REWARD:",
              reward_game, g)

    print("Time used:", time.time() - start_time)
    if PLOT_REWARD_GRAPH:
        plt.plot(reward_history)
        plt.ylabel('Reward')
        plt.xlabel('Steps')
        plt.show()
    if PLOT_TOTAL_REWARD_GRAPH:
        plt.plot(total_reward_history)
        plt.ylabel('Total game reward')
        plt.xlabel('Games')
        plt.show()
    if SAVE_NEW_MODEL:
        agent.model.save('my_new_improved_model.h5')
    print("Reward avg:", sum(reward_history) / float(len(reward_history)))
    print("Total reward avg:", sum(total_reward_history) / float(len(total_reward_history)))


if __name__ == '__main__':
    main()
