# import sys
# sys.path.append('/home/rasa/PycharmProjects/reversi-game/')

import gymnasium as gym

from stable_baselines3.common.callbacks import EvalCallback

from shutil import copyfile  # keep track of generations
from gymnasium.spaces import Discrete, Box

from game_logic import Othello
# from scripts.rl.train_model_ppo import CustomCnnPPOPolicy
import numpy as np
import os, math
from itertools import cycle


class OthelloEnv(gym.Env):
    def __init__(self, use_cnn=False):
        self.game = Othello()
        self.agent_turn = 1
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
        self.other_agent = None
        self.reset_othello_gen = self.reset_othello()
        self.episodes = 0

    def reset_othello(self):
        '''resets game to starting position
           and also changes starting player alternatively'''
        infinite_player_turn = cycle([1, 2])
        while True:
            game = Othello()
            model_turn = next(infinite_player_turn)
            yield game, model_turn

    def change_to_latest_agent(self, agent_class, agent_file_path, policy_class):
        self.other_agent = agent_class.load(agent_file_path,
                                            custom_objects={'policy_class': policy_class})

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
            if self.episodes % 500 == 0:
                print(f'Ep done - {self.episodes}.', flush=True)

            done = True
            if winner == self.agent_turn:
                reward = 1
            elif winner == 3 - self.agent_turn:  # other agent turn/figure
                reward = -1
        return reward, done

    def render(self):  # todo
        pass

    def close(self):  # todo
        pass

    def play_move(self, action):
        if isinstance(action, np.ndarray) and action.shape != ():
            action = action[0]
        game_action = Othello.get_decoded_field(action)
        self.game.play_move(game_action)

    def other_agent_play_move(self):
        obs = self.get_obs()
        action, _ = self.other_agent.predict(obs,
                                             action_masks=self.action_masks(),
                                             deterministic=False)
        self.play_move(action)

    def step(self, action):
        self.play_move(action)

        # do self play
        while self.game.get_winner() is None and self.game.player_turn != self.agent_turn:  # if game hasnt ended do moves if opponent doesnt have one
            self.other_agent_play_move()

        reward, done = self.check_game_ended()
        info = {}
        truncated = False

        # Return step information
        return self.get_obs(), reward, done, truncated, info

    def reset(self, *args, **kwargs):
        self.game, self.agent_turn = next(self.reset_othello_gen)
        if self.agent_turn == 2:
            self.other_agent_play_move()
        return self.get_obs(), None

    def action_masks(self):
        valid_moves = self.game.valid_moves()

        mask = np.zeros(self.game.board.shape, dtype=bool)

        # Set True for each index in the set
        for index in valid_moves:
            mask[index] = True
        return mask.flatten()


class OthelloEnvNoMask(OthelloEnv):
    def other_agent_play_move(self):
        obs = self.get_obs()
        action, _ = self.other_agent.predict(obs,
                                             deterministic=False)
        game_action = Othello.get_decoded_field(action)
        self.game.play_move(game_action)


class SelfPlayCallback(EvalCallback):
    # hacked it to only save new version offrom gymnasium.wrappers import FlattenObservation best model if beats prev self by BEST_THRESHOLD score
    # after saving model, resets the best score to be BEST_THRESHOLD
    def __init__(self, params, *args, **kwargs):
        super().__init__(params['eval_env'], *args, **kwargs)
        self.log_dir = params['LOGDIR']
        self.best_mean_reward = params['BEST_THRESHOLD']
        self.generation = 0
        self.params = params
        # self.train_env = params['train_env']

        # self.eval_env = params['eval_env']  # same as train env

    def set_eval(self):
        for env in self.eval_env.envs:
            env.unwrapped.set_eval()

    def set_train(self):
        for env in self.eval_env.envs:
            env.unwrapped.set_train()

    def _on_step(self) -> bool:
        # result = super()._on_step() #  eval needs to be masked, its less efficient

        result = super()._on_step()

        if result and self.best_mean_reward > self.params['BEST_THRESHOLD']:
            self.generation += 1
            print("------------------SELFPLAY: mean_reward achieved:", self.best_mean_reward,
                  '---------------------------------------')
            print("------------------SELFPLAY: new best model, bumping up generation to", self.generation,
                  '---------------------------------------', flush=True)
            source_file = os.path.join(self.log_dir, "best_model.zip")
            backup_file = os.path.join(self.log_dir, "history_" + str(self.generation).zfill(4) + ".zip")
            copyfile(source_file, backup_file)
            self.best_mean_reward = self.params['BEST_THRESHOLD']
            # agent = self.model.load(source_file)
            # agent.env = self.model.env
            # self.train_env.unwrapped.change_to_latest_agent(agent)
            self.eval_env.env_method('change_to_latest_agent',
                                     self.model.__class__,
                                     source_file,
                                     self.model.policy_class)

        return result
