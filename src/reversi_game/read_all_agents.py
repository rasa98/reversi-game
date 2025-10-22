import math
import random
import time
import numpy as np
from pathlib import Path


from reversi_game.agents.AlphaZeroAgent import (load_azero_agent)
from reversi_game.agents.MctsAgent import load_mcts_model
from reversi_game.agents.ParallelMctsAgent import load_parallel_mcts_model
from reversi_game.agents.MiniMaxAgent import (minmax_ga_best_depth_1,
                                 minmax_human_depth_1,
                                 minmax_ga_depth_dyn,
                                 minmax_human_depth_dyn,
                                 no__mmm_wpc,
                                 no_mmm,
                                 no_wpc,
                                 no__chips__corner,
                                 no__safe__danger,
                                 minmax_ga_best_depth_1_2, minmax_ga_best_depth_1_3, mmm_wpc)
from reversi_game.agents.actor_critic_agent import (load_ac_agent)
from reversi_game.agents.agent_interface import ai_random
from reversi_game.agents.sb3_agent import load_sb3_agent
from reversi_game.scripts.rl.train_model_ars import MaskableArs, CustomMlpPolicy as CustomMlpArsPolicy
from reversi_game.scripts.rl.train_model_ppo import CustomCnnPPOPolicy
from reversi_game.scripts.rl.train_model_trpo import (MaskableTrpo,
                                         CustomCnnTRPOPolicy)


def get_project_root():
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve()

    # Walk up directories until we find pyproject.toml
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    raise FileNotFoundError("Could not find project root (no pyproject.toml)")


def get_model_path(model_name):
    """Get path to model file."""
    project_root = get_project_root()
    return project_root / "models" / model_name


def load_parallel_mcts_agent_by_depth(iter_depth, c=1.41):
    mcts_params = {'max_time': math.inf,
                   'max_iter': iter_depth,
                   'c': c,
                   'verbose': 0}
    pmcts = load_parallel_mcts_model(params=mcts_params)
    return pmcts


def load_mcts_agent_by_depth(iter_depth, c=1.41):
    mcts_params = {'max_time': math.inf,
                   'max_iter': iter_depth,
                   'c': c,
                   'verbose': 0}
    mcts = load_mcts_model(params=mcts_params)
    return mcts


def load_azero_agent_with_ppo_model(iter_depth, ppo_agent, c=1.41):
    policy = ppo_agent.model.policy
    alpha_params = {'hidden_layer': 64, 'max_iter': iter_depth,
                    'dirichlet_epsilon': 0.1,
                    "uct_exploration_const": c,
                    'decay_steps': -1,
                    "final_alpha": 0.03}

    alpha = load_azero_agent(f'with ppo_model - depth {iter_depth}',
                             model=policy,
                             params=alpha_params)
    return alpha


def load_azero_agent_by_depth(iter_depth=100, azero_model_location=None, c=1.41):
    if azero_model_location is None:
        azero_model_location = get_model_path('azero.pt')
    alpha_params = {'hidden_layer': 64, 'max_iter': iter_depth,
                    'dirichlet_epsilon': 0.1,
                    "uct_exploration_const": c,
                    'decay_steps': -1,
                    "final_alpha": 0.03}

    alpha = load_azero_agent(f'{iter_depth}',
                             model_location=azero_model_location,
                             params=alpha_params)
    return alpha


# Use different seeds
seed = int(time.time())
random.seed(seed)
np.random.seed(seed)

best_mlp_ppo = load_sb3_agent(f'ppo_mlp', get_model_path('ppo_mlp'))

mcts_agent_30 = load_mcts_agent_by_depth(30)
mcts_agent_100 = load_mcts_agent_by_depth(100)
mcts_agent_500 = load_mcts_agent_by_depth(500)

pmcts_agent_30 = load_parallel_mcts_agent_by_depth(30)


azero_model_location =get_model_path('azero.pt')

alpha_100_with_ppo = load_azero_agent_with_ppo_model(100, best_mlp_ppo)
alpha_30 = load_azero_agent_by_depth(30, azero_model_location)
alpha_200 = load_azero_agent_by_depth(200, azero_model_location)

file_base_ars = get_model_path('ars_mlp')
best_ars = load_sb3_agent(f'ars_mlp',  # final1/42
                          file_base_ars,
                          cls=MaskableArs,
                          policy_cls=CustomMlpArsPolicy)

file_ppo_cnn = get_model_path('ppo_cnn')
ppo_cnn = load_sb3_agent(f'ppo_cnn',  # 69 v7
                         file_ppo_cnn,
                         cnn=True,
                         policy_cls=CustomCnnPPOPolicy)

file_base_trpo = get_model_path('trpo_cnn')
cnn_trpo = load_sb3_agent(f'trpo_cnn',  # base1 193
                          file_base_trpo,
                          MaskableTrpo,
                          cnn=True,
                          policy_cls=CustomCnnTRPOPolicy)

# ----------------------------------------
ac_agent = load_ac_agent("bare nn from azero", azero_model_location)

agents = [cnn_trpo, best_mlp_ppo,
          ppo_cnn, best_ars, minmax_ga_best_depth_1,
          minmax_human_depth_1,
          minmax_ga_depth_dyn,
          minmax_human_depth_dyn, ai_random,
          alpha_30, alpha_200,
          mcts_agent_30, mcts_agent_100, mcts_agent_500]

minmmax_agents = [ai_random,
                  minmax_human_depth_1,
                  minmax_ga_best_depth_1,
                  no_wpc,
                  no__mmm_wpc,
                  no_mmm,
                  no__safe__danger,
                  no__chips__corner,
                  minmax_ga_best_depth_1_2,
                  minmax_ga_best_depth_1_3,
                  mmm_wpc]

### Chnage so deterministic by default is False

# for agent in agents:
#     agent.set_deterministic(False)
#
# pmcts_agent_500.set_deterministic(False)
