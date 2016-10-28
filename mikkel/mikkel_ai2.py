import ast
import random
import time

import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.models import load_model

import parameters
from mikkel.mikkel_ai import evaluate_situation
from player import Player


class Poker_data:
    def __init__(self):
        self.data_states = []
        self.data_actions = []

    def next_batch(self, batch_size):
        sampled_indexes = np.random.choice(np.arange(0, len(self.data_states)), batch_size)
        batch_data = self.data_states[sampled_indexes]
        batch_actions = self.data_actions[sampled_indexes]
        return batch_data, batch_actions

    def load_data(self, file_path='data/train/'):
        self.data_states = []
        self.data_actions = []
        max = 999999999999
        counter = 0
        with open(file_path + 'inputs.txt') as f:
            content = f.readlines()
            for line in content:
                counter += 1
                x = ast.literal_eval(line)
                self.data_states.append(x)
                if counter >= max:
                    break
        counter = 0
        with open(file_path + 'answers.txt') as f:
            content = f.readlines()
            for line in content:
                counter += 1
                x = ast.literal_eval(line)
                self.data_actions.append(x)
                if counter >= max:
                    break

        self.data_states = np.array(self.data_states)
        self.data_actions = np.array(self.data_actions)
        return self.data_states, self.data_actions


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
    def __init__(self, name, model_path='mikkel/keras_models/my_model.h5'):
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
    def __init__(self, name, model_path='mikkel/keras_models/my_model.h5'):
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

    def get_bet(self, action, current_bet):
        # if action_index == 4:
        #     return self.chips
        # elif action_index == 3:
        #     return min(current_bet - self.bet + self.chips * 0.2, self.chips)
        # elif action_index == 2:
        #     return current_bet - self.bet
        # return 0
        double_bet = action * self.chips
        if action > 0.9:
            return self.chips
        elif double_bet > current_bet:
            return int(double_bet)
        elif double_bet > current_bet * 0.75:
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
