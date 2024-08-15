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
from torch import nn
import torch.nn.functional as F

from stable_baselines3.common.policies import (ActorCriticCnnPolicy,
                                               ActorCriticPolicy,
                                               MultiInputActorCriticPolicy)

# import sb3_contrib.trpo.trpo as trpo
from sb3_contrib.trpo.trpo import TRPO

from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.monitor import Monitor
from scripts.rl.env.basic_game_env import (BasicEnv,
                                           SelfPlayCallback,
                                           ReversiCNN)
from scripts.rl.env.sp_env import TrainEnv as RewardEnv

import stable_baselines3.common.callbacks as callbacks_module
from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

callbacks_module.evaluate_policy = masked_evaluate_policy
# trpo.evaluate_policy = masked_evaluate_policy

from stable_baselines3.common.utils import obs_as_tensor

th = torch


class CustomMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, obs: th.Tensor, action_masks=None, deterministic: bool = False) -> Tuple[
        th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        my_action_mask = th.tensor(action_masks, dtype=th.float32, device=self.device)

        # Neutralizing logits for invalid actions by setting them to a very low value
        distribution.distribution.logits -= (1.0 - my_action_mask) * 1e6
        
        del my_action_mask

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _predict(self, observation: th.Tensor, action_masks=None, deterministic: bool = False) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, action_masks, deterministic)
        return actions

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


class CustomCnnTRPOPolicy(CustomMlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=ReversiCNN,
                         features_extractor_kwargs=dict(features_dim=512))


class MaskableTrpo(TRPO):
    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            action_masks=None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.policy.predict(observation, state, episode_start, action_masks, deterministic)

    def collect_rollouts(
            self,
            env: VecEnv,
            callback,
            rollout_buffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)        

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                action_mask = self.env.env_method('action_masks')[0]
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_mask)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True


def get_env(env_factory, use_cnn=False):
    monitor = Monitor(env=env_factory(use_cnn))
    return DummyVecEnv([lambda: monitor])


class LinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, progress_remaining):
        return progress_remaining * self.initial_value


if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Settings
    SEED = 1141  # NOT USED
    NUM_TIMESTEPS = int(50_000_000)
    EVAL_FREQ = int(10000)
    EVAL_EPISODES = int(1000)
    BEST_THRESHOLD = 0.2  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
    # LOGDIR = 'scripts/rl/test-working/ppo/v1/'  # "ppo_masked/test/"
    LOGDIR = 'scripts/rl/output/phase2/trpo/cnn/final-A/' #'scripts/rl/output/phase2/trpo/mlp/base-v4-Rewards/'
    CNN_POLICY = True #False
    CONTINUE_FROM_MODEL = None #'scripts/rl/output/phase2/trpo/cnn/base1/history_0193' #'scripts/rl/output/phase2/trpo/cnn/base-rewards/history_0048'
    TRAIN_ENV = BasicEnv

    print(f'seed: {SEED} \nnum_timesteps: {NUM_TIMESTEPS} \neval_freq: {EVAL_FREQ}',
          f'\neval_episoded: {EVAL_EPISODES} \nbest_threshold: {BEST_THRESHOLD}',
          f'\nlogdir: {LOGDIR} \ncnn_policy: {CNN_POLICY} \ncontinueFrom_model: {CONTINUE_FROM_MODEL}', flush=True)

    policy_kwargs = {
        'net_arch': {
            'pi': [128] * 8,
            'vf': [64] * 4
        }
    }

    params = {
        'learning_rate': LinearSchedule(0.00009),
        'n_steps': 2048,#30,
        'batch_size': 128,
        #'gae_lambda': 0.95,  # Factor for GAE
        #'target_kl': 0.02,  # Maximum KL divergence between old and new policies        
        #'line_search_max_iter': 15,  # Value function coefficient        
        #'cg_max_steps': 10,  # Maximum number of conjugate gradient steps
        #'cg_damping': 0.1,  # Damping factor for conjugate gradient
        #'gamma': 0.99,
        'verbose': 100,
        'seed': SEED,
    }

    env = TRAIN_ENV
    eval_env = BasicEnv
    if CNN_POLICY:
        env = get_env(env, use_cnn=True)
        eval_env = get_env(eval_env, use_cnn=True)
        policy_class = CustomCnnTRPOPolicy
    else:
        env = get_env(env)
        eval_env = get_env(eval_env)
        policy_class = CustomMlpPolicy

    if TRAIN_ENV == BasicEnv:
        eval_env = env  # if its basic win +1/-1 reward train, use the same env for eval, cuz inner model that changes.  newer more rewards env are implemented with model playing against itself so no inner model.

    #---------------------------------------------------------------

    if CONTINUE_FROM_MODEL is None:
        params['policy_kwargs'] = policy_kwargs
        model = MaskableTrpo(policy=policy_class,
                             env=env,
                             device=device,
                             **params)
        starting_model_filepath = LOGDIR + 'random_start_model'
        model.save(starting_model_filepath)
    else:
        starting_model_filepath = CONTINUE_FROM_MODEL
        # params['exploration_rate'] = 1.0  # to reset exploration rate !!!
        model = MaskableTrpo.load(starting_model_filepath,
                                  env=env,
                                  device=device,
                                  custom_objects=params)

    print(f'\nparams: {params}\n')

    # start_model_copy = model.load(starting_model_filepath,
    #                               device=device)
    # env.envs[0].unwrapped.change_to_latest_agent(start_model_copy)
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
