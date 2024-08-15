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

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from scripts.rl.env.basic_game_env import (BasicEnv,
                                           SelfPlayCallback)
from scripts.rl.env.sp_env import TrainEnv
from scripts.rl.train_model_ppo import CustomCnnPPOPolicy


def make_env(env_cls=BasicEnv, use_cnn=False):
    def _init():
        env = env_cls(use_cnn=use_cnn)

        env2 = Monitor(env=env)
        return env2

    return _init


if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Settings
    SEED = 119  # NOT USED
    NUM_TIMESTEPS = int(200_000_000)
    N_STEPS = 2048 * 3
    EVAL_EPISODES = int(500)
    EVAL_FREQ = int(N_STEPS + 100)
    BEST_THRESHOLD = 0.18  # must achieve a mean score above this to replace prev best self
    RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
    LOGDIR = "scripts/rl/output/phase2/ppo/cnn/base-v7/"
    CNN_POLICY = True
    CONTINUE_FROM_MODEL = 'scripts/rl/output/phase2/ppo/cnn/base-v6/history_0072'  # None
    TRAIN_ENV = BasicEnv

    mp.set_start_method('forkserver')

    if os.environ['USER'] == 'student':
        num_envs = int(os.environ['SLURM_CPUS_ON_NODE']) // 2
    else:
        os.chdir('../../')
        num_envs = 2
    print(f'\nnum parallel processes: {num_envs}\n')

    

    env = TRAIN_ENV
    eval_env = BasicEnv

    if CNN_POLICY:
        env_fns = [make_env(env_cls=env, use_cnn=True) for _ in range(num_envs)]
        eval_env_fns = [make_env(env_cls=eval_env, use_cnn=True) for _ in range(num_envs)]
        policy_class = CustomCnnPPOPolicy
    else:
        env_fns = [make_env(env_cls=env) for _ in range(num_envs)]
        eval_env_fns = [make_env(env_cls=eval_env) for _ in range(num_envs)]
        policy_class = MaskableActorCriticPolicy

    vec_env = SubprocVecEnv(env_fns)

    if TRAIN_ENV == BasicEnv:
        eval_env = vec_env
    else:
        eval_env = SubprocVecEnv(eval_env_fns)

    print(f'seed: {SEED} \nnum_timesteps: {NUM_TIMESTEPS} \neval_freq: {EVAL_FREQ}',
          f'\neval_episoded: {EVAL_EPISODES} \nbest_threshold: {BEST_THRESHOLD}',
          f'\nlogdir: {LOGDIR} \ncnn_policy: {CNN_POLICY} \ncontinueFrom_model: {CONTINUE_FROM_MODEL}', flush=True)

    params = {
        'learning_rate': 3e-6,
        'n_steps': N_STEPS,
        'n_epochs': 3,
        #'clip_range': 0.20,
        'batch_size': 128,
        'ent_coef': 0.02,
        # 'gamma': 0.99,
        'verbose': 100,
        'seed': SEED,
    }

    print(f'\nparams: {params}\n')

    policy_kwargs = {
       'net_arch': {
           'pi': [128, 128] * 4,
           'vf': [64, 64] * 4
       }
    }
    # print(f'net architecture - {policy_kwargs}')

    if CONTINUE_FROM_MODEL is None:
        params['policy_kwargs'] = policy_kwargs
        model = MaskablePPO(policy=policy_class,
                            env=vec_env,
                            device=device,
                            **params)
        starting_model_filepath = LOGDIR + 'random_start_model'
        model.save(starting_model_filepath)
    else:
        starting_model_filepath = CONTINUE_FROM_MODEL
        params['policy_class'] = policy_class
        model = MaskablePPO.load(starting_model_filepath,
                                 env=vec_env,
                                 device=device,
                                 custom_objects=params)

    print(f'starting model: {starting_model_filepath}', flush=True)

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
