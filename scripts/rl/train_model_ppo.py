import os
import sys
from typing import Union, Dict, Optional, Tuple

import numpy as np
import random
from gymnasium import spaces

sys.path.append('/home/rasa/PycharmProjects/reversi-game/')
import torch
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization

import torch
import copy
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy, MultiInputPolicy

# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from scripts.rl.old_game_env import OthelloEnv, SelfPlayCallback, ReversiCNN
import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

callbacks_module.evaluate_policy = masked_evaluate_policy

th = torch


class CustomCnnPPOPolicy(MaskableActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=ReversiCNN,
                         features_extractor_kwargs=dict(features_dim=512))


def get_env(env_factory, use_cnn=False):
    monitor = Monitor(env=env_factory(use_cnn))
    return DummyVecEnv([lambda: monitor])


if __name__ == '__main__':
    # Settings
    SEED = 19  # NOT USED
    NUM_TIMESTEPS = int(30_000_000)
    EVAL_FREQ = int(20_000)
    EVAL_EPISODES = int(200)
    BEST_THRESHOLD = 0.15  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
    # LOGDIR = 'scripts/rl/test-working/ppo/v1/'  # "ppo_masked/test/"
    LOGDIR = 'scripts/rl/test-working/ppo/2cnn/'  # "ppo_masked/test/"
    CNN_POLICY = True
    CONTINUE_FROM_MODEL = None

    params = {
        'learning_rate': 0.0001,
        'n_steps': 2048 * 10,
        'n_epochs': 10,
        'clip_range': 0.15,
        'batch_size': 128,
        'ent_coef': 0.01,
        'gamma': 0.99,
        'verbose': 100,
        'seed': SEED,
    }

    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --------------------------------------------
    policy_kwargs = dict(
        net_arch=[32] * 2
    )

    env = OthelloEnv
    if CNN_POLICY:
        env = get_env(env, use_cnn=True)
        policy_class = CustomCnnPPOPolicy
    else:
        env = get_env(env)
        policy_class = MaskableActorCriticPolicy

    # starting_model_filepath = LOGDIR + 'random_start_model'
    # starting_model_filepath = 'ppo_masked/cloud/v2/history_0299'
    # starting_model_filepath = "scripts/rl/output/v3/" + 'history_0020'

    if CONTINUE_FROM_MODEL is None:
        params['policy_kwargs'] = policy_kwargs
        model = MaskablePPO(policy=policy_class,
                            env=env,
                            device=device,
                            **params)
        starting_model_filepath = LOGDIR + 'random_start_model'
        model.save(starting_model_filepath)
    else:
        starting_model_filepath = CONTINUE_FROM_MODEL
        # params['exploration_rate'] = 1.0  # to reset exploration rate !!!
        model = MaskablePPO.load(starting_model_filepath,
                                 env=env,
                                 device=device,
                                 custom_objects=params)

    start_model_copy = model.load(starting_model_filepath,
                                  device=device)
    env.envs[0].unwrapped.change_to_latest_agent(start_model_copy)

    callback_params = {
        'eval_env': env,
        'LOGDIR': LOGDIR,
        'BEST_THRESHOLD': BEST_THRESHOLD
    }

    eval_callback = SelfPlayCallback(
        callback_params,
        best_model_save_path=LOGDIR,
        log_path=LOGDIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False
    )

    model.learn(total_timesteps=NUM_TIMESTEPS,
                log_interval=100,
                callback=eval_callback)
