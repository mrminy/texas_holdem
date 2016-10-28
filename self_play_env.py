import numpy as np

import game


class self_play_env:
    """
    You need to input an agent that extends Player,
     and includes the function: get_state(game_state) and get_bet(action_index, current_bet).
    The opponent is just an agent that extends Player.
    This class would typically be used for self-play/rl to improve
    """
    def __init__(self, agent, opponent, logger=False):
        self.opponent = opponent
        self.agent = agent
        self.g = game.Texas_holdem(input_players=[agent, opponent], logger=False)
        self.state = agent.get_state(self.g.get_state())
        self.reward_history = []
        self.total_reward = 0
        self.logger = logger

        # Env
        self.action_space = np.arange(5)
        self.observation_space = np.arange(4)

    def reset(self, agent, opponent):
        self.g.reset(input_players=[agent, opponent])
        return self.agent.get_state(self.g.get_state())

    def step(self, action):
        if self.g.next_players_turn() != self.agent.id_value and len(self.g.players) > 1:
            print("ERROR!")
            return None, None, None

        bet = self.agent.get_bet(action, self.g.current_bet)
        if self.logger:
            print("Selected bet:", bet)

        reward = 0.0
        done = False
        agents_turn = False
        old_value = self.agent.chips + self.agent.bet

        # Do step and update to next state
        self.g.play_one_step(self_play=True, action=bet)
        self.state = self.agent.get_state(self.g.get_state())

        while len(self.g.players) != 1:
            if self.g.next_players_turn() != self.agent.id_value and len(self.g.players_this_round) != 1 and len(
                    self.g.players) != 1:
                # Let opponent do action
                game_state = self.g.get_state()
                bet = self.opponent.make_decision(game_state[0], game_state[1], game_state[2], game_state[3],
                                                  game_state[4], game_state[5], game_state[6])
                self.g.play_one_step(bet)
            if self.logger:
                print(agents_turn, self.g.next_players_turn(), len(self.g.players), len(self.g.players_this_round))

            out = self.g.play_one_step()
            while out == -1 and not done:
                # Is game finished?
                if len(self.g.players) == 1:
                    done = True
                else:
                    out = self.g.play_one_step()

            # Update state to next state
            self.state = self.agent.get_state(self.g.get_state())

            if self.g.next_players_turn() == self.agent.id_value:
                break

        # Calculate reward if the game is finished or this round is finished
        if len(self.g.players_this_round) == 1 or done:
            # Won the round. Calculate reward
            reward = (self.agent.chips - old_value) / 2000.0
            self.reward_history.append(reward)

        if self.logger:
            print("Round:", self.g.round_nr, "- Deal:", self.g.deal_nr, "- Reward:", reward, "- Action:", action,
                  "- Current state:", self.state, "- Done:", done)

        return self.state, reward, done, None
