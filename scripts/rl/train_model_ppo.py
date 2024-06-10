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

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from scripts.rl.game_env import OthelloEnv, OthelloEnvNoMask, SelfPlayCallback
import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

callbacks_module.evaluate_policy = masked_evaluate_policy

th = torch

# Settings
SEED = 19  # NOT USED
NUM_TIMESTEPS = int(30_000_000)
EVAL_FREQ = int(20_000)
EVAL_EPISODES = int(200)
BEST_THRESHOLD = 0.15  # must achieve a mean score above this to replace prev best self
RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
# LOGDIR = 'scripts/rl/test-working/ppo/v1/'  # "ppo_masked/test/"
LOGDIR = 'scripts/rl/test-working/ppo/1/'  # "ppo_masked/test/"

print(f'CUDA available: {torch.cuda.is_available()}')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = DummyVecEnv([lambda: Monitor(env=OthelloEnv())])

# --------------------------------------------
policy_kwargs = dict(
    net_arch=[32] * 2
)
#


# starting_model_filepath = LOGDIR + 'random_start_model'
starting_model_filepath = 'ppo_masked/cloud/v2/history_0299'
# starting_model_filepath = "scripts/rl/output/v3/" + 'history_0020'


# model = MaskablePPO(MaskableActorCriticPolicy,
#                     env=env,
#                     device=device,
#                     learning_rate=0.0001,
#                     n_steps=2048 * 10,
#                     n_epochs=10,
#                     clip_range=0.15,
#                     batch_size=128,
#                     ent_coef=0.01,
#                     gamma=0.99,
#                     verbose=100,
#                     seed=SEED
#                     )
params = {'learning_rate': 0.00007,
          'n_steps': 2048 * 10,
          'n_epochs': 10,
          'clip_range': 0.3,
          'batch_size': 128,
          'ent_coef': 0.05,
          'gamma':0.96}

model = MaskablePPO.load(starting_model_filepath, env=env, custom_objects=params)

print(f'device {model.device}')

# model.save(starting_model_filepath)
start_model_copy = model.load(starting_model_filepath)


env.envs[0].unwrapped.change_to_latest_agent(start_model_copy)

params = {
    'eval_env': env,
    'LOGDIR': LOGDIR,
    'BEST_THRESHOLD': BEST_THRESHOLD
}

eval_callback = SelfPlayCallback(
    params,
    best_model_save_path=LOGDIR,
    log_path=LOGDIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=EVAL_EPISODES,
    deterministic=False
)

model.learn(total_timesteps=NUM_TIMESTEPS,
            log_interval=100,
            callback=eval_callback)
