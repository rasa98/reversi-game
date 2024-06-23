import os
import sys
from typing import Union, Dict, Optional, Tuple

import numpy as np
import random
from gymnasium import spaces

if __name__ == '__main__' and os.environ['USER'] != 'student':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
    sys.path.append(source_dir)
    os.chdir('../../')

import torch
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization

import torch
from torch import nn
import torch.nn.functional as F

from sb3_contrib.ars.policies import LinearPolicy, MlpPolicy
import sb3_contrib.ars.ars as ars

from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.monitor import Monitor
from scripts.rl.old_game_env import OthelloEnv, SelfPlayCallback

import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

callbacks_module.evaluate_policy = masked_evaluate_policy
ars.evaluate_policy = masked_evaluate_policy

th = torch


class CustomLinearPolicy(LinearPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        if isinstance(self.action_space, spaces.Box):
            raise Exception('Action masking only for discrete action space !!!')
        # return self.action_net(features)
        elif isinstance(self.action_space, spaces.Discrete):
            return self.action_net(features)
            # logits = self.action_net(features)
            # return th.argmax(logits, dim=1)
        else:
            raise NotImplementedError()

    def _predict(self, observation: th.Tensor, action_masks=None, deterministic: bool = False) -> th.Tensor:
        if action_masks is None:
            raise Exception("Needs to have actio masks!!")
        logits = self.forward(observation)
        action_masks = th.tensor(action_masks, dtype=th.float32, device=self.device)
        masked_logits = logits + (1.0 - action_masks) * -10e6
        if deterministic:
            return th.argmax(masked_logits, dim=1)
        else:
            probabilities = th.softmax(masked_logits, dim=1)
            return th.multinomial(probabilities, num_samples=1).squeeze(1)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            action_masks=None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, action_masks=action_masks, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]


class CustomMlpPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        if isinstance(self.action_space, spaces.Box):
            raise Exception('Action masking only for discrete action space !!!')
        # return self.action_net(features)
        elif isinstance(self.action_space, spaces.Discrete):
            return self.action_net(features)
            # logits = self.action_net(features)
            # return th.argmax(logits, dim=1)
        else:
            raise NotImplementedError()

    def _predict(self, observation: th.Tensor, action_masks=None, deterministic: bool = False) -> th.Tensor:
        if action_masks is None:
            raise Exception("Needs to have actio masks!!")
        logits = self.forward(observation)
        action_masks = th.tensor(action_masks, dtype=th.float32, device=self.device)
        masked_logits = logits + (1.0 - action_masks) * -10e6
        if deterministic:
            return th.argmax(masked_logits, dim=1)
        else:
            probabilities = th.softmax(masked_logits, dim=1)
            return th.multinomial(probabilities, num_samples=1).squeeze(1)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            action_masks=None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, action_masks=action_masks, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc, assignment]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low,
                                  self.action_space.high)  # type: ignore[assignment, arg-type]

        # Remove batch dimension if needed
        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]


class MaskableArs(ars.ARS):
    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            action_masks=None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, action_masks, deterministic)


class LinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, progress_remaining):
        return progress_remaining * self.initial_value


def get_env(env_factory):
    monitor = Monitor(env=env_factory())
    return DummyVecEnv([lambda: monitor])


if __name__ == '__main__':
    # Settings
    SEED = 129  # NOT USED
    NUM_TIMESTEPS = int(10_000_000)
    EVAL_FREQ = int(26000)
    EVAL_EPISODES = int(200)
    BEST_THRESHOLD = 0.3  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.    
    LOGDIR = 'scripts/rl/output/phase2/ars/mlp/base-v2/'  # "ppo_masked/test/"
    CONTINUE_FROM_MODEL = None #"scripts/rl/output/phase2/ars/mlp/base/history_0140"

    print(f'seed: {SEED} \nnum_timesteps: {NUM_TIMESTEPS} \neval_freq: {EVAL_FREQ}',
          f'\neval_episoded: {EVAL_EPISODES} \nbest_threshold: {BEST_THRESHOLD}',
          f'\nlogdir: {LOGDIR} \ncontinueFrom_model: {CONTINUE_FROM_MODEL}', flush=True)

    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = OthelloEnv
    env = get_env(env)

    eval_env = env

    policy_kwargs = dict(
        net_arch=[64] * 8
    )

    params = {
        'n_delta': 30,
        'n_top': 8,
        'zero_policy': False,
        'n_eval_episodes': 20,
        'delta_std': 0.05,
        'learning_rate': LinearSchedule(5e-3),
        'verbose': 1,
        'seed': SEED,
    }

    if CONTINUE_FROM_MODEL is None:
        params['policy_kwargs'] = policy_kwargs
        model = MaskableArs(policy=CustomMlpPolicy,
                            env=env,
                            device=device,
                            **params)
        starting_model_filepath = LOGDIR + 'random_start_model'
        model.save(starting_model_filepath)
    else:
        starting_model_filepath = CONTINUE_FROM_MODEL
        # params['exploration_rate'] = 1.0  # to reset exploration rate !!!
        params['policy_class'] = CustomMlpPolicy
        model = MaskableArs.load(starting_model_filepath,
                                 env=env,
                                 device=device,
                                 custom_objects=params)

    print(f'\nparams: {params}\n')

    #start_model_copy = model.load(starting_model_filepath, custom_objects={'policy_class': CustomMlpPolicy})
    #env.envs[0].unwrapped.change_to_latest_agent(start_model_copy)
    eval_env.env_method('change_to_latest_agent',
                        model.__class__,
                        starting_model_filepath,
                        model.policy_class)

    params = {
        'eval_env': eval_env,
        'LOGDIR': LOGDIR,
        'BEST_THRESHOLD': BEST_THRESHOLD
    }

    eval_callback = SelfPlayCallback(
        params,
        best_model_save_path=LOGDIR,
        log_path=LOGDIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True
    )

    model.learn(total_timesteps=NUM_TIMESTEPS,
                log_interval=1,
                callback=eval_callback)
