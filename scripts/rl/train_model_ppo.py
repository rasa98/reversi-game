import os
import sys
from typing import Union, Dict, Optional, Tuple

import numpy as np
import random
from gymnasium import spaces

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

if os.environ['USER'] == 'rasa':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
    sys.path.append(source_dir)

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from scripts.rl.env.old_game_env import (BasicEnv,
                                         SelfPlayCallback,
                                         ReversiCNN)
import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

callbacks_module.evaluate_policy = masked_evaluate_policy

th = torch


class LinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, progress_remaining):
        return progress_remaining * self.initial_value


class CustomCnnPPOPolicy(MaskableActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=ReversiCNN,
                         features_extractor_kwargs=dict(features_dim=512))


def get_env(env_factory, use_cnn=False):
    monitor = Monitor(env=env_factory(use_cnn))
    return DummyVecEnv([lambda: monitor])


if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Settings
    SEED = 12  # NOT USED
    NUM_TIMESTEPS = int(12_000_000)
    EVAL_FREQ = int(20_500 * 3)
    EVAL_EPISODES = int(200)
    BEST_THRESHOLD = 0.22  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
    # LOGDIR = 'scripts/rl/test-working/ppo/v1/'  # "ppo_masked/test/"
    LOGDIR = 'scripts/rl/output/phase2/ppo/cnn/base-v3/'  # "ppo_masked/test/"
    CNN_POLICY = True
    CONTINUE_FROM_MODEL = None #'scripts/rl/output/phase2/ppo/cnn/base-v3/history_0024' #None


    print(f'seed: {SEED} \nnum_timesteps: {NUM_TIMESTEPS} \neval_freq: {EVAL_FREQ}',
          f'\neval_episoded: {EVAL_EPISODES} \nbest_threshold: {BEST_THRESHOLD}',
          f'\nlogdir: {LOGDIR} \ncnn_policy: {CNN_POLICY} \ncontinueFrom_model: {CONTINUE_FROM_MODEL}', flush=True)

    params = {
        'learning_rate': LinearSchedule(5e-5),
        'n_steps': 2048 * 30,
        'n_epochs': 5,
        'clip_range': 0.25,
        'batch_size': 128,
        'ent_coef': 0.015,
        #'gamma': 0.99,
        'verbose': 100,
        'seed': SEED,
    }

    print(f'\nparams: {params}\n')

    # --------------------------------------------
    policy_kwargs = {
        'net_arch': {
            'pi': [128, 128] * 4,
            'vf': [64, 64] * 4
        }
    }

    env = BasicEnv
    if CNN_POLICY:
        env = get_env(env, use_cnn=True)
        policy_class = CustomCnnPPOPolicy
    else:
        env = get_env(env)
        policy_class = MaskableActorCriticPolicy

    eval_env = env

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
        params['policy_class'] = CustomCnnPPOPolicy  # trained on different version libs...
        model = MaskablePPO.load(starting_model_filepath,
                                 env=env,
                                 device=device,
                                 custom_objects=params)

    # start_model_copy = model.load(starting_model_filepath,
    #                               device=device)
    # env.envs[0].unwrapped.change_to_latest_agent(start_model_copy)
    eval_env.env_method('change_to_latest_agent',
                        model.__class__,
                        starting_model_filepath,
                        model.policy_class)

    callback_params = {
        'eval_env': eval_env,
        'LOGDIR': LOGDIR,
        'BEST_THRESHOLD': BEST_THRESHOLD
    }

    eval_callback = SelfPlayCallback(
        callback_params,
        best_model_save_path=LOGDIR,
        log_path=LOGDIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True
    )

    model.learn(total_timesteps=NUM_TIMESTEPS,
                log_interval=1,
                callback=eval_callback)
