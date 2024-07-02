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


X_SQUARES = {(1, 1), (6, 6), (1, 6), (6, 1)}
C_SQUARES = {(0, 2), (0, 5), (7, 2), (7, 5), (2, 0), (2, 7), (5, 0), (5, 7)}


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
        if winner is not None:
            self.episodes += 1
            if self.episodes % 1000 == 0:
                print(f'Ep done - {self.episodes}.')

            done = True
            if winner == self.game.last_turn:
                reward = 100
            elif winner == 3 - self.game.last_turn:  # other agent turn/figure
                reward = -100
        else:  # if not done, get some other rewards
            if last_played_move in self.game.CORNERS:
                reward += 10
            elif self.game.turn < 20:
                if last_played_move in X_SQUARES:
                    reward -= 5
                elif last_played_move in C_SQUARES:
                    reward -= 3


            ### give solidly big reward if you play move that will make you again the next player
            if self.game.player_turn == self.game.last_turn:  # oplayer turn is always here the agent_turn
                reward += 10
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

