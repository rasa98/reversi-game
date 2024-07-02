import random
import time
import math
import numpy as np

from agents.MiniMaxAgent import (load_minimax_agent,
                                 mm_static,
                                 mm2_dynamic,
                                 ga_0,
                                 ga_1,
                                 ga_2,
                                 ga_vpn_5,
                                 ga_new,
                                 ga_human,
                                 ga2_best)
from agents.sb3_agent import load_sb3_model
from agents.agent_interface import ai_random
from agents.MctsAgent import load_mcts_model
from agents.ParallelMctsAgent import load_parallel_mcts_model

from agents.AlphaZeroAgent import (load_azero_model,
                                   multi_folder_load_models,
                                   multi_folder_load_some_models)

from bench_agent import bench_both_sides

from scripts.rl.train_model_ars import MaskableArs, CustomMlpPolicy as CustomMlpArsPolicy
from scripts.rl.train_model_dqn import MaskableDQN
from scripts.rl.train_model_trpo import (MaskableTrpo,
                                         CustomMlpPolicy as CustomMlpTrpoPolicy,
                                         CustomCnnTRPOPolicy)

from scripts.rl.train_model_ppo import CustomCnnPPOPolicy
from sb3_contrib.ppo_mask import MaskablePPO


def load_mcts_agent_by_depth(iter_depth, c=1.41):
    mcts_params = {'max_time': math.inf,
                   'max_iter': iter_depth,
                   'c': c,
                   'verbose': 0}
    mcts = load_mcts_model(params=mcts_params)
    return mcts


def load_azero_agent_by_depth(iter_depth, azero_model_location, c=1.41):
    alpha_params = {'hidden_layer': 64, 'max_iter': iter_depth,
                    'dirichlet_epsilon': 0.1,
                    "uct_exploration_const": c,
                    'decay_steps': -1,
                    "final_alpha": 0.03}

    alpha = load_azero_model(f'depth {iter_depth}',
                             file=azero_model_location,
                             params=alpha_params)
    return alpha


if __name__ == '__main__':
    # Use different seeds
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    best_mlp_ppo = load_sb3_model(f'ppo_mlp', 'models/ppo_mlp')

    mcts_agent_30 = load_mcts_agent_by_depth(30)
    mcts_agent_200 = load_mcts_agent_by_depth(200)
    mcts_agent_500 = load_mcts_agent_by_depth(500)

    azero_folder = 'models/'  # f'models_output/alpha-zero/FINAL/layer64-LAST-v3/'
    azero_model_location = f'{azero_folder}azero.pt'  # 3

    alpha_30 = load_azero_agent_by_depth(30, azero_model_location)
    alpha_200 = load_azero_agent_by_depth(200, azero_model_location)

    file_base_ars = 'models/ars_mlp'
    best_ars = load_sb3_model(f'ars 201',
                              file_base_ars,
                              cls=MaskableArs,
                              policy_cls=CustomMlpArsPolicy)

    file_ppo_cnn = 'models/ppo_cnn'
    ppo_cnn = load_sb3_model(f'ppo_cnn 19',
                             file_ppo_cnn,
                             cnn=True,
                             policy_cls=CustomCnnPPOPolicy)

    file_base_trpo = 'models/trpo_cnn'
    cnn_trpo = load_sb3_model(f'trpo_cnn 48',
                              file_base_trpo,
                              MaskableTrpo,
                              cnn=True,
                              policy_cls=CustomCnnTRPOPolicy)

    # ----------------------------------------

    from elo_rating import Tournament as Tour

    agents = [cnn_trpo, best_mlp_ppo, ppo_cnn, best_ars, ga_human, ga2_best, ai_random, alpha_30, alpha_200,
              mcts_agent_30, mcts_agent_200, mcts_agent_500]
    [agent.set_deterministic(False) for agent in agents]

    log_dir = 'demo elo rating 2 with dirichlet'
    rounds = 100
    verbose = 0
    t = Tour(agents, log_dir, rounds=rounds, verbose=verbose)
    start = time.perf_counter()
    t.simulate()
    end = time.perf_counter()
    print(f'Time needed: {end - start} secs')
