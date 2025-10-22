import copy
import os
import sys
from functools import partial
from typing import Union, Dict, Optional, Tuple, List

import numpy as np
from gymnasium import spaces
from sb3_contrib.common.vec_env import AsyncEval

# if __name__ == '__main__' and (os.environ.get('USER') or os.environ.get('USERNAME')) != 'student':
#     source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
#     sys.path.append(source_dir)
#     os.chdir('../../reversi_game/')

from stable_baselines3.common.vec_env import DummyVecEnv

import torch

from sb3_contrib.ars.policies import LinearPolicy, MlpPolicy
import sb3_contrib.ars.ars as ars

from stable_baselines3.common.monitor import Monitor
from reversi_game.scripts.rl.env.basic_game_env import BasicEnv, SelfPlayCallback

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
    def __init__(self, eval_env=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_env = eval_env

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

    def evaluate_candidates(
            self, candidate_weights: th.Tensor,
            callback: callbacks_module.BaseCallback,
            async_eval: Optional[AsyncEval]
    ) -> th.Tensor:
        """
        Evaluate each candidate.

        :param candidate_weights: The candidate weights to be evaluated.
        :param callback: Callback that will be called at each step
            (or after evaluation in the multiprocess version)
        :param async_eval: The object for asynchronous evaluation of candidates.
        :return: The episodic return for each candidate.
        """

        batch_steps = 0
        # returns == sum of rewards
        candidate_returns = th.zeros(self.pop_size, device=self.device)
        train_policy = copy.deepcopy(self.policy)
        # Empty buffer to show only mean over one iteration (one set of candidates) in the logs
        self.ep_info_buffer = []
        callback.on_rollout_start()

        if async_eval is not None:
            # Multiprocess asynchronous version
            async_eval.send_jobs(candidate_weights, self.pop_size)
            results = async_eval.get_results()

            for weights_idx, (episode_rewards, episode_lengths) in results:
                # Update reward to cancel out alive bonus if needed
                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                batch_steps += np.sum(episode_lengths)
                self._mimic_monitor_wrapper(episode_rewards, episode_lengths)

            # Combine the filter stats of each process for normalization
            for worker_obs_rms in async_eval.get_obs_rms():
                if self._vec_normalize_env is not None:
                    # worker_obs_rms.count -= self.old_count
                    self._vec_normalize_env.obs_rms.combine(worker_obs_rms)
                    # Hack: don't count timesteps twice (between the two are synced)
                    # otherwise it will lead to overflow,
                    # in practice we would need two RunningMeanStats
                    self._vec_normalize_env.obs_rms.count -= self.old_count

            # Synchronise VecNormalize if needed
            if self._vec_normalize_env is not None:
                async_eval.sync_obs_rms(self._vec_normalize_env.obs_rms.copy())
                self.old_count = self._vec_normalize_env.obs_rms.count

            # Hack to have Callback events
            for _ in range(batch_steps // len(async_eval.remotes)):
                self.num_timesteps += len(async_eval.remotes)
                callback.on_step()
        else:
            # Single process, synchronous version
            for weights_idx in range(self.pop_size):
                # Load current candidate weights
                train_policy.load_from_vector(candidate_weights[weights_idx].cpu())
                # Evaluate the candidate
                episode_rewards, episode_lengths = masked_evaluate_policy(
                    train_policy,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    return_episode_rewards=True,
                    # Increment num_timesteps too (slight mismatch with multi envs)
                    callback=partial(self._trigger_callback, callback=callback, n_envs=self.eval_env.num_envs),
                    warn=False,
                )
                # Update reward to cancel out alive bonus if needed
                candidate_returns[weights_idx] = sum(episode_rewards) + self.alive_bonus_offset * sum(episode_lengths)
                batch_steps += sum(episode_lengths)
                self._mimic_monitor_wrapper(episode_rewards, episode_lengths)

            # Note: we increment the num_timesteps inside the evaluate_policy()
            # however when using multiple environments, there will be a slight
            # mismatch between the number of timesteps used and the number
            # of calls to the step() method (cf. implementation of evaluate_policy())
            # self.num_timesteps += batch_steps

        callback.on_rollout_end()

        return candidate_returns

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "eval_env",
            "replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]


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
    SEED = 141  # NOT USED
    NUM_TIMESTEPS = int(200_000_000)
    EVAL_FREQ = int(1000000)
    EVAL_EPISODES = int(1000)
    BEST_THRESHOLD = 0.3  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
    LOGDIR = 'scripts/rl/output/phase2/ars/mlp/FINAL-B-1/'  # "ppo_masked/test/"
    CONTINUE_FROM_MODEL = None #'scripts/rl/output/phase2/ars/mlp/base-new/history_0183'  # None
    TRAIN_ENV = BasicEnv # SelfPlayEnv


    print(f'seed: {SEED} \nnum_timesteps: {NUM_TIMESTEPS} \neval_freq: {EVAL_FREQ}',
          f'\neval_episoded: {EVAL_EPISODES} \nbest_threshold: {BEST_THRESHOLD}',
          f'\nlogdir: {LOGDIR} \ncontinueFrom_model: {CONTINUE_FROM_MODEL}', flush=True)

    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = TRAIN_ENV
    eval_env = BasicEnv

    env = get_env(env)
    eval_env = get_env(eval_env)

    if TRAIN_ENV == BasicEnv:
        eval_env = env  # if its basic win +1/-1 reward train, use the same env for eval, cuz inner model that changes.  newer more rewards env are implemented with model playing against itself so no inner model.

    policy_kwargs = dict(
        net_arch=[128] * 8
    )

    params = {
        'n_delta': 30,
        'n_top': 3,
        'zero_policy': False,
        'n_eval_episodes': 400,
        # 'delta_std': 0.03,
        'learning_rate': LinearSchedule(2e-2),
        'verbose': 1,
        'seed': SEED,
    }

    if CONTINUE_FROM_MODEL is None:
        params['policy_kwargs'] = policy_kwargs
        model = MaskableArs(eval_env=eval_env,
                            policy=CustomMlpPolicy,
                            env=env,
                            device=device,
                            **params)
        starting_model_filepath = LOGDIR + 'random_start_model'
        model.save(starting_model_filepath)
    else:
        starting_model_filepath = CONTINUE_FROM_MODEL

        params['policy_class'] = CustomMlpPolicy
        model = MaskableArs.load(starting_model_filepath,
                                 env=env,
                                 device=device,
                                 custom_objects=params)
        model.eval_env = eval_env

    print(f'\nparams: {params}\n')

    # start_model_copy = model.load(starting_model_filepath, custom_objects={'policy_class': CustomMlpPolicy})
    # env.envs[0].unwrapped.change_to_latest_agent(start_model_copy)
    eval_env.env_method('change_to_latest_agent',
                        model.__class__,
                        starting_model_filepath,
                        model.policy_class)
    if TRAIN_ENV != BasicEnv:
        env.env_method('change_to_latest_agent',
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
