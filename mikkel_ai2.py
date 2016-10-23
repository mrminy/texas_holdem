import ast
import random
import time
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.models import load_model

import parameters
from mikkel_ai import evaluate_situation
from player import Player


class Poker_data:
    def __init__(self):
        self.data_states = []
        self.data_actions = []

    def next_batch(self, batch_size):
        batch_data = []
        batch_actions = []
        for k in range(batch_size):
            id = random.randint(0, len(self.data_states) - 1)
            batch_data.append(self.data_states[id])
            batch_actions.append(self.data_actions[id])
        return batch_data, batch_actions

    def load_data(self, file_path='data/train/'):
        with open(file_path + 'inputs.txt') as f:
            content = f.readlines()
            for line in content:
                x = ast.literal_eval(line)
                self.data_states.append(x)

        with open(file_path + 'answers.txt') as f:
            content = f.readlines()
            for line in content:
                x = ast.literal_eval(line)
                self.data_actions.append(x)

        return np.array(self.data_states), np.array(self.data_actions)


def train_tf_model():
    batch_size = 95
    epochs = 50
    validation_split = 0.15
    nb_classes = 5
    nb_inputs = 4

    print("Loading data...")
    data_train = Poker_data()
    data_train.load_data('data/train/')
    data_test = Poker_data()
    data_test.load_data('data/test/')

    # Build model
    layer_sizes = [4, 20, 20, 20, 5]
    batch_size = 95
    hm_epochs = 50
    validation_split = 0.15

    x = tf.placeholder('float', [None, layer_sizes[0]])
    y = tf.placeholder('float')

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[0], layer_sizes[1]])),
                      'biases': tf.Variable(tf.random_normal([layer_sizes[1]]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[1], layer_sizes[2]])),
                      'biases': tf.Variable(tf.random_normal([layer_sizes[2]]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[2], layer_sizes[3]])),
                      'biases': tf.Variable(tf.random_normal([layer_sizes[3]]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([layer_sizes[3], layer_sizes[4]])),
                    'biases': tf.Variable(tf.random_normal([layer_sizes[4]]))}

    l1 = tf.add(tf.matmul(data_train.data_states, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, data_train.data_actions))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Train and test model
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_cost = 0
            for _ in range(int(len(data_train.data_states) / batch_size)):
                epoch_x, epoch_y = data_train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_cost += c
            print('Epoch', epoch, 'completed of', hm_epochs, 'loss', epoch_cost)
        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: data_test.data_states, y: data_test.data_actions}))

        # Save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, 'my_tf_model.ckpt')
        print('Model saved in file: %s' % save_path)


def train_model():
    batch_size = 95
    epochs = 50
    validation_split = 0.15
    nb_classes = 5
    nb_inputs = 4

    print("Loading data...")
    data_train = Poker_data()
    data_train.load_data('data/train/')
    data_test = Poker_data()
    data_test.load_data('data/test/')

    print("Creating model...")
    model = Sequential()
    model.add(Dense(20, input_dim=nb_inputs))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.25))
    model.add(Dense(20))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.25))
    model.add(Dense(20))
    model.add(Activation('tanh'))
    # model.add(Dropout(0.25))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print("Training data...")
    # tb = TensorBoard(log_dir='./logs/relu_20_20_20')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    model.fit(data_train.data_states, data_train.data_actions, batch_size=batch_size, nb_epoch=epochs, verbose=1,
              validation_split=validation_split, callbacks=[])
    score = model.evaluate(data_test.data_states, data_test.data_actions, batch_size=batch_size, verbose=1)
    print("Final score:", score)

    # Uncomment to save model
    model.save('my_model.h5')


class My_Keras_SL_AI(Player):
    def __init__(self, name, model_path='my_model.h5'):
        super().__init__(name)
        self.model = load_model(model_path)

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board, round_nr):
        start_time = time.time()

        win_rate, loose_rate, tie_rate = evaluate_situation(parameters.EVALUATION_PRECISION,
                                                            len(players_this_round) - 1, self.hand, board)

        if current_bet - max(0, self.bet) != 0:
            what_i_have_to_bet = current_bet - self.bet
            what_im_been_offered = pot
            pot_odds = what_im_been_offered / what_i_have_to_bet
        else:
            pot_odds = 0.0

        state = [win_rate, loose_rate, tie_rate, pot_odds]
        prediction = self.model.predict_proba(np.array(state).reshape(1, 4), batch_size=1, verbose=0)
        pred = prediction[0].reshape(5)
        max_index = np.argmax(pred)

        # print("State:", state)
        # print("Action:", max_index)
        if max_index == 4:
            return self.chips
        elif max_index == 3:
            return min(current_bet - self.bet + self.chips * 0.2, self.chips)
        elif max_index == 2:
            return current_bet - self.bet
        return 0


class My_Keras_SL_AI_Self_Play_Learner(Player):
    def __init__(self, name, model_path='my_model.h5'):
        super().__init__(name)
        self.model = load_model(model_path)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    def make_decision(self, betting_history, current_bet, max_bet, players_this_round, pot, board, round_nr):
        start_time = time.time()

        state = self.get_state([betting_history, current_bet, max_bet, players_this_round, pot, board, round_nr])
        prediction = self.model.predict_proba(np.array(state).reshape(1, 4), batch_size=1, verbose=0)
        pred = prediction[0].reshape(5)
        return self.get_bet(np.argmax(pred), current_bet)

    def get_state(self, game_state):
        win_rate, loose_rate, tie_rate = evaluate_situation(parameters.EVALUATION_PRECISION,
                                                            len(game_state[3]) - 1, self.hand, game_state[5])
        if game_state[1] - max(0, self.bet) != 0:
            what_i_have_to_bet = game_state[1] - self.bet
            what_im_been_offered = game_state[4]
            pot_odds = what_im_been_offered / what_i_have_to_bet
        else:
            pot_odds = 0.0
        return [win_rate, loose_rate, tie_rate, pot_odds]

    def get_bet(self, action_index, current_bet):
        if action_index == 4:
            return self.chips
        elif action_index == 3:
            return min(current_bet - self.bet + self.chips * 0.2, self.chips)
        elif action_index == 2:
            return current_bet - self.bet
        return 0

    def get_action_index(self, current_bet, bet):
        if bet == self.chips:
            return 4
        elif bet == min(current_bet - self.bet + self.chips * 0.2, self.chips):
            return 3
        elif bet == current_bet - self.bet:
            return 2
        return 0

    def get_action(self, current_state):
        random_float = random.random()
        if random_float <= parameters.EPSILON:
            # Select random action
            action_index = random.randint(0, 4)
            return self.get_bet(action_index, current_state[1]), action_index
        # Select action from policy
        bet = self.make_decision(current_state[0], current_state[1], current_state[2], current_state[3],
                                 current_state[4], current_state[5], current_state[6])
        return bet, self.get_action_index(current_state[1], bet)


if __name__ == '__main__':
    train_model()
