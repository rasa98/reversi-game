import random
import time
import math
import numpy as np

from models.MiniMaxAgent import (load_minimax_agent,
                                 mm_static,
                                 mm2_dynamic,
                                 ga_0,
                                 ga_1,
                                 ga_2,
                                 ga_vpn_5,
                                 ga_new,
                                 ga_human,
                                 ga2_best)
from models.sb3_model import load_sb3_model
from models.model_interface import ai_random
from models.MctsModel import load_mcts_model
from models.ParallelMctsModel import load_parallel_mcts_model

from models.AlphaZeroModel import (load_azero_model,
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

    best_mlp_ppo = load_sb3_model(f'ppo_mlp', 'scripts/rl/output/paral/v3v3-1/history_0032')

    mcts_agent_30 = load_mcts_agent_by_depth(30)
    mcts_agent_200 = load_mcts_agent_by_depth(200)
    mcts_agent_500 = load_mcts_agent_by_depth(500)

    azero_folder = 'models_output/alpha-zero/FINAL/layer64-LAST-v4/'  # f'models_output/alpha-zero/FINAL/layer64-LAST-v3/'
    azero_model_location = f'{azero_folder}model_4.pt'  # 3

    alpha_30 = load_azero_agent_by_depth(30, azero_model_location)
    alpha_200 = load_azero_agent_by_depth(200, azero_model_location)

    file_base_ars = 'scripts/rl/output/phase2/ars/mlp/base-new/history_0201'
    best_ars = load_sb3_model(f'ars 201',
                              file_base_ars,
                              cls=MaskableArs,
                              policy_cls=CustomMlpArsPolicy)

    file_ppo_cnn = 'scripts/rl/output/phase2/ppo/cnn/base-v5/history_0019'
    ppo_cnn = load_sb3_model(f'ppo_cnn 19',
                             file_ppo_cnn,
                             cnn=True,
                             policy_cls=CustomCnnPPOPolicy)

    file_base_trpo = 'scripts/rl/output/phase2/trpo/cnn/base-rewards/history_0048'
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
