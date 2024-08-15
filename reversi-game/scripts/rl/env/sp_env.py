# import sys
# sys.path.append('/home/rasa/PycharmProjects/reversi-game/')

import gymnasium as gym
import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch as th
from torch import nn

# Modify the namespace of EvalCallback directly
# callbacks_module.evaluate_policy = masked_evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback

from shutil import copyfile  # keep track of generations
from gymnasium.spaces import Discrete, Box, MultiBinary

from game_logic import Othello
import numpy as np
import os, math
from itertools import cycle


WEIGHT_MATRIX = np.array([
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [5,    -2, -1,  0,  0, -1,  -2,   5],
    [5,    -2, -1,  0,  0, -1,  -2,   5],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100]
])

WEIGHT_MATRIX_2 = np.array([
    [100, -20, -10, -5, -5, -10, -20, 100],
    [-20, -50,  -2, -2, -2,  -2, -50, -20],
    [-10,  -2,  -1, -1, -1,  -1,  -2, -10],
    [-5,   -2,  -1,  0,  0,  -1,  -2,  -5],
    [-5,   -2,  -1,  0,  0,  -1,  -2,  -5],
    [-10,  -2,  -1, -1, -1,  -1,  -2, -10],
    [-20, -50,  -2, -2, -2,  -2, -50, -20],
    [100, -20, -10, -5, -5, -10, -20, 100]
])


class TrainEnv(gym.Env):
    def __init__(self, use_cnn=False):
        self.game = Othello()
        # self.agent_turn = 1
        shape = self.game.board.shape
        self.action_space = Discrete(shape[0] * shape[1])  # sample - [x, y]
        # self.observation_space = Dict({
        #                                 'board' : Box(0, 2, shape=shape, dtype=int),
        #                                 'player': Discrete(2, start=1)
        #                               })
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

    def check_game_ended(self, last_played_move):
        reward = 0
        done = False
        winner = self.game.get_winner()
        turn = self.game.turn
        if winner is not None:
            self.episodes += 1
            if self.episodes % 1000 == 0:
                print(f'Ep done - {self.episodes}.')

            done = True
            if winner == self.game.last_turn:
                reward = 200
            elif winner == 3 - self.game.last_turn:  # other agent turn/figure
                reward = -200
        else:  # if not done, get some other rewards
            rew = WEIGHT_MATRIX[last_played_move]
            ratio = (60 - turn) / 60
            reward = ratio * rew
        return reward, done

    def render(self):  # todo
        pass

    def close(self):  # todo
        pass

    def step(self, action):
        game_action = Othello.get_decoded_field(action)
        self.game.play_move(game_action)

        reward, done = self.check_game_ended(game_action)
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

