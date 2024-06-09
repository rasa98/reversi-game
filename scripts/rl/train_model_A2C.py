import os
import sys
from typing import Union, Dict, Optional, Tuple

import numpy as np
import random
from gymnasium import spaces

sys.path.append('/home/rasa/PycharmProjects/reversi-game/')
import torch

from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, sync_envs_normalization
from stable_baselines3 import A2C

import torch
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


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict(self, observation, action_masks=None, deterministic=False):
        # Call the parent class to get the logits
        distribution = super()._predict(observation, deterministic)

        # Apply the action mask
        valid_actions = action_masks
        logits = distribution.distribution.logits
        logits[~valid_actions] = float('-inf')

        # Create a new distribution with the masked logits
        masked_distribution = CategoricalDistribution(self.action_space.n)
        masked_distribution.distribution = torch.distributions.Categorical(logits=logits)

        return masked_distribution


class CustomDQNPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict(self, observation, action_masks=None, deterministic=False):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not isinstance(action_masks, torch.Tensor):
            action_masks = torch.tensor(action_masks, dtype=torch.bool, device=self.device)
            if action_masks.ndim == 1:
                action_masks = action_masks.unsqueeze(0)
        q_values = self.q_net(observation)
        # Apply action mask
        q_values[~action_masks] = float('-inf')

        if deterministic:
            value = torch.argmax(q_values, dim=-1)
        else:
            dist = CategoricalDistribution(q_values.size(-1))
            dist.proba_distribution(q_values)
            value = dist.sample()

        if action_masks.ndim == 1:
            return value.item(), None
        return value.detach().numpy(), None


class MaskableA2C(A2C):
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        pass


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
                action_masks = self.env.action_masks()  # 1-dim
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                res = []
                for i in range(n_batch):
                    env = self.env.envs[i].unwrapped
                    random_move = random.choice(list(env.game.valid_moves()))
                    encoded_random_move = env.game.get_encoded_field(random_move)
                    res.append(encoded_random_move)
                action = np.array(res)
            else:
                env = self.env
                random_move = random.choice(list(env.game.valid_moves()))
                encoded_random_move = env.game.get_encoded_field(random_move)
                action = np.array(encoded_random_move)
        else:
            action, state = self.policy._predict(observation, action_masks=action_masks, deterministic=deterministic)
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


def get_env(env_factory):
    monitor = Monitor(env=env_factory())
    return DummyVecEnv([lambda: monitor])


# Settings
SEED = 19  # NOT USED
NUM_TIMESTEPS = int(30_000_000)
EVAL_FREQ = int(30_000)
EVAL_EPISODES = int(400)
BEST_THRESHOLD = 0.3  # must achieve a mean score above this to replace prev best self
RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
LOGDIR = 'scripts/rl/test-working/ppo/v1/'  # "ppo_masked/test/"
NON_MASKABLE_MODEL = False

print(f'CUDA available: {torch.cuda.is_available()}')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if not NON_MASKABLE_MODEL:
    import stable_baselines3.common.callbacks as callbacks_module
    from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

    callbacks_module.evaluate_policy = masked_evaluate_policy
    env = OthelloEnv
else:
    env = OthelloEnvNoMask

env = get_env(env)
# env = lambda: Monitor(env=env())
# env = DummyVecEnv([lambda: env()])

# model = MaskablePPO(policy=MaskableActorCriticPolicy,
#                     env=env,
#                     device=device,
#                     verbose=1)
# --------------------------------------------
# policy_kwargs = dict(
#     net_arch=[256, 128, 128, 64]
# )
#
# model = MaskableDQN(policy=CustomDQNPolicy,
#                     env=env,
#                     device=device,
#                     learning_rate=1e-2,
#                     # buffer_size=100000,  # 1e5
#                     # learning_starts=50000,
#                     # batch_size=128,
#                     # tau=0.99,
#                     # gamma=0.99,
#                     # train_freq=15,
#                     # gradient_steps=1,
#                     # target_update_interval=15000,
#                     # exploration_fraction=0.05,
#                     # exploration_initial_eps=1.0,
#                     # exploration_final_eps=0.05,
#                     verbose=1,
#                     policy_kwargs=policy_kwargs)
#--------------------------------------------------
import os
print(os.getcwd())
starting_model_filepath = 'ppo_masked/cloud/v2/history_0299'  # LOGDIR + 'random_start_model'
model = MaskablePPO.load(starting_model_filepath, env=env, device=device,
                         learning_rate=0.0001,
                         n_steps=2048*2,
                         clip_range=0.15,
                         batch_size=128,
                         ent_coef=0.01,
                         gamma=0.99,
                         )
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
