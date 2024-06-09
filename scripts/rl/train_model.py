import torch
from stable_baselines3.common.vec_env import DummyVecEnv, sync_envs_normalization

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
from scripts.rl.ppo_masked import OthelloEnv, SelfPlayCallback


# Settings
SEED = 19  # NOT USED
NUM_TIMESTEPS = int(50_000_000)
EVAL_FREQ = int(2048 * 20 + 1)
EVAL_EPISODES = int(400)
BEST_THRESHOLD = 0.20  # must achieve a mean score above this to replace prev best self
RENDER_MODE = False  # set this to false if you plan on running for full 1000 trials.
LOGDIR = "scripts/rl/output/v3/"

env = OthelloEnv()
env = Monitor(env=env)

print(f'CUDA available: {torch.cuda.is_available()}')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

policy_kwargs = {
    'net_arch': {
        'pi': [128, 128, 128, 128], 
        'vf': [128, 128, 128, 128] 
    }
}
print(f'net architecture - {policy_kwargs}')

print(f'params: \nNUM_TIMESTEPS -{NUM_TIMESTEPS}\nEVAL_FREQ={EVAL_FREQ}\nEVAL_EPISODES={EVAL_EPISODES}\nBEST_THRESHOLD={BEST_THRESHOLD}\nLOGDIR={LOGDIR}')

params = {
    'learning_rate': 0.0001,
    'n_steps': 2048 * 20,
    'n_epochs': 10,
    'clip_range': 0.15,
    'batch_size': 128,
    'ent_coef': 0.01,
    'gamma': 0.99,
    'verbose': 1
}

print(f'model params: \n {params}')

model = MaskablePPO(policy=MaskableActorCriticPolicy,
                    env=env,
                    device=device,                    
                    policy_kwargs=policy_kwargs,
                    **params)
starting_model_filepath = LOGDIR + 'random_start_model'
# model = MaskablePPO.load(starting_model_filepath, env=env)
model.save(starting_model_filepath)

start_model_copy = model.load(starting_model_filepath)
env.unwrapped.change_to_latest_agent(start_model_copy)

env_eval = OthelloEnv()
env_eval = Monitor(env=env_eval)

env_eval = DummyVecEnv(env_fns=[lambda: env_eval])
env_eval.envs[0].unwrapped.change_to_latest_agent(start_model_copy)
params = {
    'train_env': env,
    'eval_env': env_eval,
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

model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)
