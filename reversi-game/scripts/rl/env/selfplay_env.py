# import sys
# sys.path.append('/home/rasa/PycharmProjects/reversi-game/')

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

from game_logic import Othello

# Modify the namespace of EvalCallback directly
# callbacks_module.evaluate_policy = masked_evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback


X_SQUARES = {(1, 1), (6, 6), (1, 6), (6, 1)}
C_SQUARES = {(0, 2), (0, 5), (7, 2), (7, 5), (2, 0), (2, 7), (5, 0), (5, 7)}


class SelfPlayEnv(gym.Env):
    def __init__(self, use_cnn=False):
        self.game = Othello()
        shape = self.game.board.shape
        self.action_space = Discrete(shape[0] * shape[1])  # sample - [x, y]
        self.use_cnn = use_cnn
        if use_cnn:
            self.observation_space = Box(low=0, high=255, shape=(3, 8, 8), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=1, shape=(64 * 3,), dtype=np.float32)
        self.episodes = 0

    def get_obs(self):
        if self.use_cnn:
            encoded_board = self.game.get_encoded_state_as_img()
        else:
            encoded_board = self.game.get_encoded_state().reshape(-1)
        return encoded_board

    def check_game_ended(self):
        reward = 0
        done = False
        winner = self.game.get_winner()
        if winner is not None:
            self.episodes += 1
            done = True
            if winner == self.game.last_turn:
                reward = 1
            elif winner == 3 - self.game.last_turn:  # other agent turn/figure
                reward = -1
        return reward, done

    def render(self):  # todo
        pass

    def close(self):  # todo
        pass

    def step(self, action):
        game_action = Othello.get_decoded_field(action)
        self.game.play_move(game_action)

        reward, done = self.check_game_ended()
        info = {}
        truncated = False

        # Return step information
        return self.get_obs(), reward, done, truncated, info

    def reset(self, *args, **kwargs):
        self.game = Othello()
        return self.get_obs(), None

    def action_masks(self):
        valid_moves = self.game.valid_moves()
        mask = np.zeros(self.game.board.shape, dtype=bool)

        # Set True for each index in the set
        for index in valid_moves:
            mask[index] = True
        return mask.flatten()

