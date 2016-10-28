import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from marius.marius_ai import ai as marius_ai
from mikkel.mikkel_ai import My_Experimenter_AI2 as mikkel_ai

import self_play_env
from johannes.johannes_ai import pokerAI as johannes_ai
from mikkel.mikkel_ai2 import My_Keras_SL_AI_Self_Play_Learner
from player import Call_player

# [-0.5, 0.005, 0.5075, 0.6045, 0.1475, -0.5, 0.5, 0.429, -0.22, -0.1275, -0.49, 0.5, 0.6565, 0.0845, 0.125, 0.515, -0.1415, -0.47, -0.0015, -0.248, -0.005, -0.122, -0.001, -0.0015, -0.076, -0.115, -0.0025, -0.0015, 0.02, -0.036, -0.001, -0.004, -0.005, 0.133, -0.006, -0.0015, 0.3655, -0.0005, 0.499, -0.0035, -0.0035, -0.0015, -0.006, -0.001, -0.024, -0.013, -0.0035, -0.0015, -0.0015, -0.2535, -0.0035, -0.0705, -0.001, -0.0045, -0.004, -0.0015, -0.002, -0.0005, -0.0035, -0.0045, -0.001, -0.0035, -0.0005, -0.0025, -0.001, -0.0015, -0.16, -0.0035, -0.006, -0.001, -0.005, -0.5, -0.008, -0.0135, -0.0005, -0.0015, -0.0045, -0.0015, -0.0025, -0.001, -0.0055, 0.131, -0.001, -0.006, -0.0015, -0.0015, 0.1155, -0.005, -0.4575, -0.0055, -0.0045, -0.0005, -0.004, -0.006, -0.002, -0.002, -0.0055, 0.005, -0.212, -0.3335, -0.165, -0.002, -0.002, -0.004, -0.0065, -0.0035, -0.0015, -0.0045, 0.0575, 0.788, -0.208, -0.0025, -0.0015, -0.0075, -0.027, -0.0055, -0.004, -0.435, -0.005, -0.41, -0.001, -0.0055, -0.006, -0.0045, -0.0005, -0.0035, -0.004, -0.003, -0.0035, -0.002, -0.005, -0.005, -0.0015, -0.007, -0.002, -0.0045, -0.0015, -0.005, -0.07, -0.001, -0.0045, -0.001, -0.003, -0.0045, -0.0045, -0.0045, -0.0025, -0.0005, -0.0015, 0.1075, 0.0825, -0.0025, -0.0045, -0.0045, -0.001, -0.0015, 0.482, -0.004, -0.4765, -0.11, -0.0015, -0.0015, -0.005, -0.0035, -0.49, -0.36, -0.001, -0.0035, -0.129, -0.0005, -0.004, -0.006, -0.005, 0.6045, -0.0015, 0.0845, -0.0025, -0.0045, -0.0015, -0.0035, -0.005, -0.003, -0.005, -0.0015, -0.0045, -0.0015, -0.001, -0.14, 0.788, -0.001, -0.0065, 0.109, -0.1275, -0.002, -0.003, -0.003, -0.0025, -0.006, -0.0015, -0.002, -0.004, -0.0045, -0.006, -0.0025, -0.323, -0.001, -0.006, -0.005, -0.0015, -0.002, -0.0015, -0.1525, -0.0045, -0.006, -0.004, -0.0115, 0.475, -0.003, -0.0055, -0.0005, -0.0035, -0.0015, -0.0005, -0.1315, -0.005, -0.005, -0.1375, -0.006, -0.136, -0.0045, -0.0005, -0.003, -0.005, -0.258, 0.609, -0.002, -0.1735, -0.002, -0.3955, -0.0035, -0.0015, -0.0005, -0.0005, -0.0075, -0.0045, 0.1315, -0.395, -0.0705, -0.0035, -0.004]

# HYPERPARMETERS
H = 200
H2 = 200
batch_number = 500
gamma = 0.99
num_between_q_copies = 150
explore_decay = 0.9999
min_explore = 0.02
memory_size = 1000000
learning_rate = 0.00005


class DQN:
    def __init__(self, env, logger=False):
        # Set up the environment
        self.env = env
        self.logger = logger

        self.graph = tf.Graph()
        self.all_assigns = None
        self.Q = None
        self.Q_ = None
        self.optimize = None
        self.init = None
        self.target_q = None
        self.states_ = None
        self.action_used = None
        self.saver = None
        self.sess = None

        self.D = []
        self.build()

    def build(self):
        # First Q Network
        with self.graph.as_default():
            w1 = tf.Variable(tf.random_uniform([self.env.observation_space.shape[0], H], -1.0, 1.0))
            b1 = tf.Variable(tf.random_uniform([H], -1.0, 1.0))

            w2 = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
            b2 = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

            w3 = tf.Variable(tf.random_uniform([H2, len(self.env.action_space)], -1.0, 1.0))
            b3 = tf.Variable(tf.random_uniform([len(self.env.action_space)], -1.0, 1.0))

            # Second Q Network
            w1_ = tf.Variable(tf.random_uniform([self.env.observation_space.shape[0], H], -1.0, 1.0))
            b1_ = tf.Variable(tf.random_uniform([H], -1.0, 1.0))

            w2_ = tf.Variable(tf.random_uniform([H, H2], -1.0, 1.0))
            b2_ = tf.Variable(tf.random_uniform([H2], -1.0, 1.0))

            w3_ = tf.Variable(tf.random_uniform([H2, len(self.env.action_space)], -1.0, 1.0))
            b3_ = tf.Variable(tf.random_uniform([len(self.env.action_space)], -1.0, 1.0))

            # Make assign functions for updating Q prime's weights
            w1_update = w1_.assign(w1)
            b1_update = b1_.assign(b1)
            w2_update = w2_.assign(w2)
            b2_update = b2_.assign(b2)
            w3_update = w3_.assign(w3)
            b3_update = b3_.assign(b3)

            self.all_assigns = [
                w1_update,
                w2_update,
                w3_update,
                b1_update,
                b2_update,
                b3_update]

            # build network
            self.states_ = tf.placeholder(tf.float32, [None, self.env.observation_space.shape[0]])
            h_1 = tf.nn.relu(tf.matmul(self.states_, w1) + b1)
            h_2 = tf.nn.relu(tf.matmul(h_1, w2) + b2)
            h_2 = tf.nn.dropout(h_2, .5)
            self.Q = tf.nn.softmax(tf.matmul(h_2, w3) + b3)

            h_1_ = tf.nn.relu(tf.matmul(self.states_, w1_) + b1_)
            h_2_ = tf.nn.relu(tf.matmul(h_1_, w2_) + b2_)
            h_2_ = tf.nn.dropout(h_2_, .5)
            self.Q_ = tf.nn.softmax(tf.matmul(h_2_, w3_) + b3_)

            self.action_used = tf.placeholder(tf.int32, [None], name="action_masks")
            action_masks = tf.one_hot(self.action_used, len(self.env.action_space))
            filtered_Q = tf.reduce_sum(tf.mul(self.Q, action_masks), reduction_indices=1)

            # Train Q
            self.target_q = tf.placeholder(tf.float32, [None, ])
            loss = tf.reduce_mean(tf.square(filtered_Q - self.target_q))
            self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            self.saver = tf.train.Saver()

            self.init = tf.initialize_all_variables()

            self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.sess.run(self.all_assigns)

    def play_games(self, n_games):
        # self.sess.run(self.init)
        # self.sess.run(self.all_assigns)
        winners = np.zeros(2)
        tot_ticks = 0
        for episode in range(n_games):
            state = self.env.reset(env.agent, env.opponent)

            while True:
                tot_ticks += 1
                q = self.sess.run(self.Q, feed_dict={self.states_: np.array([state])})[0]
                # action = np.argmax(q)
                new_state, reward, done, _ = self.env.step(q[0])

                if done:
                    break
            if self.env.g.players[0].id_value == agent.id_value:
                winners[0] += 1
            else:
                winners[1] += 1
        return winners

    def learn(self, max_episodes, save=False):
        explore = 1.0
        reward_list = []
        recent_rewards = []
        past_actions = []
        episode_number = 0

        ticks = 0
        for episode in range(max_episodes):
            state = self.env.reset(env.agent, env.opponent)

            reward_sum = 0.0

            while True:
                ticks += 1

                if episode % 10 == 0:
                    q, qp = self.sess.run([self.Q, self.Q_], feed_dict={self.states_: np.array([state])})
                    if self.logger:
                        print("Q:{}, Q_ {}".format(q[0], qp[0]))

                if explore > random.random():
                    action = random.random()  # np.random.choice(len(self.env.action_space))
                else:
                    q = self.sess.run(self.Q, feed_dict={self.states_: np.array([state])})[0]
                    # action = np.argmax(q)
                    action = q[0]
                explore = max(explore * explore_decay, min_explore)

                new_state, reward, done, _ = self.env.step(action)
                reward_sum += reward

                self.D.append([state, action, reward, new_state, done])
                if len(self.D) > memory_size:
                    self.D.pop(0)

                state = new_state

                if done:
                    break

                # Training a Batch
                samples = random.sample(self.D, min(batch_number, len(self.D)))
                if len(samples) > 0:
                    # calculate all next Q's together
                    new_states = [x[3] for x in samples]
                    all_q = self.sess.run(self.Q_, feed_dict={self.states_: new_states})

                    y_ = []
                    state_samples = []
                    actions = []
                    terminalcount = 0
                    for ind, i_sample in enumerate(samples):
                        state_mem, curr_action, reward, new_state, done = i_sample
                        if done:
                            y_.append(reward)
                            terminalcount += 1
                        else:
                            this_q = all_q[ind]
                            maxq = max(this_q)
                            y_.append(reward + (gamma * maxq))

                        state_samples.append(state_mem)

                        actions.append(curr_action)
                    self.sess.run([self.optimize],
                                  feed_dict={self.states_: state_samples, self.target_q: y_, self.action_used: actions})
                    if ticks % num_between_q_copies == 0:
                        self.sess.run(self.all_assigns)

            reward_list.append(reward_sum)
            recent_rewards.append(reward_sum)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            print('Reward for episode %d is %d. Explore is %.4f' % (episode, reward_sum, explore))
        if save:
            self.saver.save(self.sess, 'dqn_self_play_model.ckpt')

        # print(reward_list)
        print("Total nr of episodes:", episode_number)
        print(reward_list)

        plt.plot(reward_list)
        plt.show()


def fetch_opponent():
    """
    :return: a randomly selected opponent
    """
    possible_opponents = [Call_player("Call player"), mikkel_ai("Mikkel AI"), johannes_ai("Johannes AI"),
                          marius_ai("Marius AI")]
    # My_Keras_SL_AI("Old Keras AI", model_path='data/old_models/my_model_relu_20_20_dropout.h5')
    return random.choice(possible_opponents)


if __name__ == '__main__':
    agent = My_Keras_SL_AI_Self_Play_Learner("New Keras AI", model_path='my_model.h5')
    opponent = Call_player("Call player")
    env = self_play_env.self_play_env(agent, opponent, action_size=2)
    dqn = DQN(env)
    # dqn.restore(path='self_play_agent/sl_models/first_tf_model_for_dqn.ckpt')

    # Test against opponent n times
    n_games = 500
    winners_1 = dqn.play_games(n_games)

    dqn.learn(max_episodes=2000)

    # Test against opponent n times
    winners_2 = dqn.play_games(n_games)

    print(winners_1)
    print(winners_2)
