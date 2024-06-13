import torch
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, sync_envs_normalization
import os, sys

if os.environ['USER'] == 'rasa':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
    sys.path.append(source_dir)

import stable_baselines3.common.callbacks as callbacks_module

from sb3_contrib.common.maskable.evaluation import evaluate_policy as masked_evaluate_policy

# Modify the namespace of EvalCallback directly
callbacks_module.evaluate_policy = masked_evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from scripts.rl.game_env import OthelloEnv, SelfPlayCallback


# Settings
SEED = 19  # NOT USED
NUM_TIMESTEPS = int(100_000_000)
N_STEPS = 512 * 30
EVAL_FREQ = int(N_STEPS + 1)
EVAL_EPISODES = int(100)
BEST_THRESHOLD = 0.3  # must achieve a mean score above this to replace prev best self
RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
LOGDIR = "scripts/rl/output/v1/"


class LinearSchedule:
    def __init__(self, initial_value):
        self.initial_value = initial_value

    def __call__(self, progress_remaining):
        return progress_remaining * self.initial_value


def make_env():
    def _init():
        env = OthelloEnv()
        env2 = Monitor(env=env)
        return env2

    return _init


if __name__ == '__main__':
    mp.set_start_method('forkserver')
    num_envs = 1
    env_fns = [make_env() for _ in range(num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    # env = OthelloEnv()
    # env = Monitor(env=env)

    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    policy_kwargs = {
        'net_arch': {
            'pi': [64, 64],
            'vf': [64, 64]
        }
    }
    print(f'net architecture - {policy_kwargs}')

    print(
        f'params: \nNUM_TIMESTEPS -{NUM_TIMESTEPS}\nEVAL_FREQ={EVAL_FREQ}\nEVAL_EPISODES={EVAL_EPISODES}\nBEST_THRESHOLD={BEST_THRESHOLD}\nLOGDIR={LOGDIR}')

    params = {
        'learning_rate': 9e-5,  #LinearSchedule(9e-5),
        'n_steps': N_STEPS,
        'n_epochs': 5,
        # 'clip_range': 0.2,
        'batch_size': 64,
        # 'ent_coef': 0.01,
        # 'gae_lambda': 0.95,
        # 'gamma': 1,
        # 'verbose': 1
    }

    print(f'model params: \n {params}')

    model = MaskablePPO(policy=MaskableActorCriticPolicy,
                        env=vec_env,
                        device=device,
                        policy_kwargs=policy_kwargs,
                        **params)

    starting_model_filepath = LOGDIR + 'random_start_model'
    model.save(starting_model_filepath)

    #  ------ load pretrained ---------
    # starting_model_filepath = "scripts/rl/output/v3/" + 'history_0020'
    # model = MaskablePPO.load(starting_model_filepath, env=vec_env, custom_objects=params)

    print(f'starting model: {starting_model_filepath}')

    vec_env.env_method('change_to_latest_agent', model.__class__, starting_model_filepath)


    # env_eval = OthelloEnv()
    # env_eval = Monitor(env=env_eval)
    # env_eval = DummyVecEnv(env_fns=[lambda: env_eval])
    #
    # start_model_copy = model.load(starting_model_filepath)
    # env_eval.envs[0].unwrapped.change_to_latest_agent(start_model_copy)
    params = {
        'eval_env': vec_env,
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

    model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)
