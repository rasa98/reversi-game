import os
import sys
from typing import Union, Dict, Optional, Tuple

import numpy as np
import random
from gymnasium import spaces

sys.path.append('/home/rasa/PycharmProjects/reversi-game/')

from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization

import torch
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy, CnnPolicy, MultiInputPolicy

# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from scripts.rl.env.old_game_env import (BasicEnv,
                                         SelfPlayCallback,
                                         ReversiCNN)

import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

callbacks_module.evaluate_policy = masked_evaluate_policy

th = torch


class CustomDQNPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        with torch.no_grad():
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

    def _predict(self, obs, action_masks=None, deterministic=False):
        self = self.q_net
        q_values = self.q_net(self.extract_features(obs, self.features_extractor))

        # action_masks = th.tensor(action_masks, dtype=th.float32, device=self.device)

        # Apply action mask

        q_values[~action_masks] = float('-inf')
        # q_values = q_values + (1.0 - action_masks) * -10e6  # when action mask is not tensor

        return th.argmax(q_values, dim=1)
        # if deterministic:
        #     return th.argmax(q_values, dim=1)
        # else:
        #     probabilities = th.softmax(q_values, dim=1)
        #     return th.multinomial(probabilities, num_samples=1).squeeze(1)


class CustomCnnDQNPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=ReversiCNN,
                         features_extractor_kwargs=dict(features_dim=512))

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

        with torch.no_grad():
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

    def _predict(self, obs, action_masks=None, deterministic=False):
        self = self.q_net
        q_values = self.q_net(self.extract_features(obs, self.features_extractor))

        # action_masks = th.tensor(action_masks, dtype=th.float32, device=self.device)

        # Apply action mask

        q_values[~action_masks] = float('-inf')
        # q_values = q_values + (1.0 - action_masks) * -10e6  # when action mask is not tensor

        return th.argmax(q_values, dim=1)
        # if deterministic:
        #     return th.argmax(q_values, dim=1)
        # else:
        #     probabilities = th.softmax(q_values, dim=1)
        #     return th.multinomial(probabilities, num_samples=1).squeeze(1)


class MaskableDQN(DQN):
    def predict(
            self,
            observation,
            state=None,
            episode_start=None,
            action_masks=None,
            deterministic=False,
    ):
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if action_masks is None:
            if isinstance(self.env, VecEnv):
                action_masks = np.stack(self.env.env_method('action_masks'))  # 2-dim, batch of masks
            else:
                raise Exception('Need to wrap it in VecENV !!!!')

        action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
        if action_masks_tensor.ndim == 1:
            action_masks_tensor = action_masks_tensor.unsqueeze(0)

        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                res = []
                for i in range(n_batch):
                    # env = self.env.envs[i].unwrapped
                    valid_moves_batch_list = [np.where(mask)[0].tolist() for mask in action_masks]
                    random_move_list = [random.choice(valid_moves) for valid_moves in valid_moves_batch_list]
                    # encoded_random_move = env.game.get_encoded_field(random_move)
                    res += random_move_list
                action = np.array(res)
            else:
                res = []
                # env = self.env
                valid_moves = np.where(action_masks)[0].tolist()
                random_move = random.choice(valid_moves)
                # encoded_random_move = env.game.get_encoded_field(random_move)
                res.append(random_move)
                action = np.array(res)
        else:
            action, state = self.policy.predict(observation,
                                                action_masks=action_masks_tensor,
                                                deterministic=deterministic)
        return action, state

    def _sample_action(
            self,
            learning_starts: int,
            action_noise=None,
            n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            res = []
            for i in range(n_envs):
                env = self.env.envs[i].unwrapped
                random_move = random.choice(list(env.game.valid_moves()))
                encoded_random_move = env.game.get_encoded_field(random_move)
                res.append(encoded_random_move)
            unscaled_action = np.array(res)
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


def get_env(env_factory, use_cnn=False):
    monitor = Monitor(env=env_factory(use_cnn))
    return DummyVecEnv([lambda: monitor])


if __name__ == '__main__':

    # Settings
    SEED = 11  # NOT USED
    NUM_TIMESTEPS = int(30_000_000)
    EVAL_FREQ = int(160_00)
    EVAL_EPISODES = int(50)
    BEST_THRESHOLD = 0.25  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
    # LOGDIR = 'scripts/rl/test-working/ppo/v1/'  # "ppo_masked/test/"
    LOGDIR = 'scripts/rl/test-working/dqn/5cnn/'  # "ppo_masked/test/"
    CNN_POLICY = True
    CONTINUE_FROM_MODEL = None  # 'scripts/rl/test-working/dqn/4/history_0004'
    TRAIN_ENV = BasicEnv

    policy_kwargs = dict(
        net_arch=[64] * 4
    )

    params = {
        'learning_rate': 1e-4,
        'buffer_size': 160_00,
        'learning_starts': 1,  # train_freq + learning_starts = first train
        'batch_size': 128,
        # tau=0.5,
        # gamma=0.9,
        'train_freq': (50, "episode"),
        # gradient_steps=1,
        # max_grad_norm=20,  # 10
        # target_update_interval=10000,
        'exploration_fraction': 0.7,
        'exploration_initial_eps': 0.9,
        'exploration_final_eps': 0.3,
        'verbose': 1,
        'seed': SEED
    }

    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = TRAIN_ENV
    eval_env = BasicEnv
    if CNN_POLICY:
        env = get_env(env, use_cnn=True)
        eval_env = get_env(eval_env, use_cnn=True)
        policy_class = CustomCnnDQNPolicy
    else:
        env = get_env(env)
        eval_env = get_env(eval_env)
        policy_class = CustomDQNPolicy

    if TRAIN_ENV == BasicEnv:
        eval_env = env  # if its basic win +1/-1 reward train, use the same env for eval, cuz inner model that changes.  newer more rewards env are implemented with model playing against itself so no inner model.

    # --------------------------------------------
    #
    if CONTINUE_FROM_MODEL is None:
        params['policy_kwargs'] = policy_kwargs
        model = MaskableDQN(policy=policy_class,
                            env=env,
                            device=device,
                            **params)
        starting_model_filepath = LOGDIR + 'random_start_model'
        model.save(starting_model_filepath)
    else:
        starting_model_filepath = CONTINUE_FROM_MODEL
        params['exploration_rate'] = 1.0  # to reset exploration rate !!!
        model = MaskableDQN.load(starting_model_filepath,
                                 env=env,
                                 device=device,
                                 custom_objects=params)

    # start_model_copy = model.load(starting_model_filepath,
    #                               device=device,
    #                               custom_objects=params)
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
                log_interval=260,
                callback=eval_callback)
