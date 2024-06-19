import random
import time

from models.minmax import (mm_static,
                           mm2_dynamic,
                           ga_0,
                           ga_1,
                           ga_2,
                           ga_vpn_5)
from models.ppo_masked_model import (load_model_new,
                                     )
from models.model_interface import ai_random
from models.montecarlo import mcts_model
from models.ParallelMCTS import PMCTS
import models.AlphaZero
from models.AlphaZeroModel import (load_azero_model,
                                   multi_folder_load_models,
                                   multi_folder_load_some_models)
# from models.AlphaZero import (multi_folder_load_some_models)
from bench_agent import bench_both_sides

from cProfile import Profile
from pstats import SortKey, Stats


def ppo_death_match(multi_ppo, times=100):
    multi_ppo = list(set(multi_ppo))  # shuffle it
    i = 1
    best = multi_ppo[0]
    for idx, ppo in enumerate(multi_ppo[i:], i):  # many_zero_models:
        ai1_wins, ai2_wins = bench_both_sides(best, ppo, times=times)
        more_wins = ai1_wins - ai2_wins
        if more_wins <= 0:
            best = multi_ppo[idx]
    print(f'Best ppo is: {best.name}')


if __name__ == '__main__':
    ppo_299 = load_model_new('cloud_299', 'scripts/rl/ppo_masked/cloud/v2/history_0299.zip')  # file)
    ppo_113 = load_model_new('cloud_113', 'scripts/rl/ppo_masked/cloud/v2/history_0113.zip')  # file)
    ppo_531 = load_model_new('cloud_531', 'scripts/rl/ppo_masked/cloud/v2/history_0531.zip')
    ppo_del = load_model_new('cloud_test', 'scripts/rl/scripts/rl/test-working/ppo/v1/history_0004.zip')  # file)
    ppo_del2 = load_model_new('cloud_test2', 'scripts/rl/scripts/rl/test-working/ppo/v1/history_0003.zip')  # file)
    # ppo_18_big_rollouts = load_model_new('18 big rollout', 'scripts/rl/scripts/rl/test-working/ppo/1/history_0018')

    cloud_random = load_model_new(f'ppo_random_cloud', f'scripts/rl/ppo_masked/cloud/paral/random_start_model')
    file_base = 'scripts/rl/ppo_masked/cloud/paral/history_'
    multi_ppo = (load_model_new(f'ppo_{i}', f'{file_base}{str(i).zfill(4)}')
                 for i in range(14, 17))

    file_base_v3v3 = 'scripts/rl/output/v3v3/history_'
    multi_ppo_v3v3 = lambda: (load_model_new(f'ppo_{i}', f'{file_base_v3v3}{str(i).zfill(4)}')
                              for i in range(17, 33))
    it = multi_ppo_v3v3()
    best_ppo_yet = next(it)
    best_next_18 = next(it)

    file_base_v3v3v1 = 'scripts/rl/output/v3v3-1/history_'
    multi_ppo_v3v3v1 = lambda: (load_model_new(f'ppo_{i}', f'{file_base_v3v3v1}{str(i).zfill(4)}')
                                for i in [24, 32, 33, 37])  # 32 univerzalno bolji

    file_base_v4 = 'scripts/rl/output/v4/history_'
    multi_ppo_v4 = lambda: (load_model_new(f'ppo_{i}', f'{file_base_v4}{str(i).zfill(4)}')
                            for i in range(42, 49))

    file_base = 'scripts/rl/output/paral/base/v0/history_'
    multi_ppo_paral_v0 = (load_model_new(f'ppo_{i}', f'{file_base}{str(i).zfill(4)}')
                          for i in [8, 9, 10])

    file_base = 'scripts/rl/output/paral/base/v1/history_'
    multi_ppo_paral_v1 = (load_model_new(f'ppo_{i}', f'{file_base}{str(i).zfill(4)}')
                          for i in [1, 2, 3, 4, 5])

    file_base = 'scripts/rl/output/paral/base/v1.1/history_'
    multi_ppo_paral_v11 = (load_model_new(f'ppo_{i}', f'{file_base}{str(i).zfill(4)}')
                           for i in [1, 2, 3, 4, 5])

    # file_base = 'scripts/rl/scripts/rl/test-working/ppo/1/history_'
    # multi_ppo = (load_model_new(f'ppo_{i}_bigg rollouts', f'{file_base}{str(i).zfill(4)}')
    #              for i in range(14, 19))

    pmcts = PMCTS('parallel mcts',
                  time_limit=1,
                  iter_limit=1000
                  )

    azero_folder = f'models_output/alpha-zero/FINAL2/res4layer256-v1/'
    azero_model_location = f'{azero_folder}model_0.pt'
    alpha_params = {'res_blocks': 4, 'hidden_layer': 256, 'max_iter': 300,
                    'dirichlet_epsilon': 0.1, "uct_exploration_const": 1.41,
                    "final_alpha": 0.1}

    alpha = load_azero_model(f'model 0 256',
                             file=azero_model_location,
                             params=alpha_params)

    azero_folder_params = [(azero_folder, range(2, 4), alpha_params)]
    many_zero_models = lambda: multi_folder_load_some_models(azero_folder_params)

    # with PMCTS.create_pool_manager(pmcts, num_processes=4):
    #     bench_both_sides(
    #                      pmcts,
    #                      # mcts_model,
    #                      alpha_329,
    #                      times=5)

    # bench_both_sides(ppo_18_big_rollouts,
    #                  ppo_299,
    #                  # alpha_mcts,
    #                  # mcts_model,
    #                  # ga_vpn_5,
    #                  # ga_vpn_5,
    #                  times=100)

    import random
    import numpy as np
    import time

    # Use different seeds
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    # for agent in [cloud_random] + list(multi_ppo):
    # with PMCTS.create_pool_manager(pmcts, num_processes=4):
    #     for agent in list(multi_ppo_v3v3v1):
    #         bench_both_sides(
    #             best_next_18.set_deterministic(False),
    #             #best_ppo_yet.set_deterministic(False),
    #             # ga_vpn_5,
    #             agent.set_deterministic(False),
    #             # ppo_del2.set_deterministic(False),#agent,
    #             times=50,
    #             timed=True,
    #             verbose=1)
    # with PMCTS.create_pool_manager(pmcts, num_processes=4):

    from scripts.rl.train_model_ars import MaskableArs
    from scripts.rl.train_model_dqn import MaskableDQN
    from scripts.rl.train_model_trpo import MaskableTrpo, CustomMlpPolicy as CustomMlpTrpoPolicy

    from scripts.rl.train_model_ppo import CustomCnnPPOPolicy
    from sb3_contrib.ppo_mask import MaskablePPO

    file_base_ars = 'scripts/rl/scripts/rl/test-working/ars/del2/history_'
    multi_ars = lambda: (load_model_new(f'ars_{i}',
                                        f'{file_base_ars}{str(i).zfill(4)}',
                                        MaskableArs)
                         for i in range(14, 34))  # num 15 je najbolji

    file_base_dqn = 'scripts/rl/scripts/rl/test-working/dqn/4v1/history_'
    multi_dqn = lambda: (load_model_new(f'dqn_{i}',
                                        f'{file_base_dqn}{str(i).zfill(4)}',
                                        MaskableDQN)
                         for i in range(1, 75))  # num 15 je najbolji

    file_base_ppo_cnn = 'scripts/rl/scripts/rl/test-working/ppo/1/history_'
    multi_ppo_cnn_paral_v0 = lambda: (load_model_new(f'ppo_{i}',
                                                     f'{file_base}{str(i).zfill(4)}',
                                                     cnn=True)
                                      for i in [1, 2, 3, 4])

    file_base_trpo_cnn = 'scripts/rl/scripts/rl/test-working/trpo/test/history_'
    multi_trpo_cnn = lambda: (load_model_new(f'trpo_{i}',
                                             f'{file_base_trpo_cnn}{str(i).zfill(4)}',
                                             MaskableTrpo,
                                             cnn=True)
                              for i in range(1, 9))

    file_ppo_base2_cnn = 'scripts/rl/ppo_masked/cnn/base2/history_'
    ppo_base2_cnn = lambda: (load_model_new(f'ppo_cnn{i}',
                                            f'{file_ppo_base2_cnn}{str(i).zfill(4)}',
                                            cnn=True,
                                            policy_cls=CustomCnnPPOPolicy)
                             for i in [62])  # range(1, 52))

    file_base_trpo = 'scripts/rl/trpo/mlp/first run/history_'
    multi_trpo = lambda: (load_model_new(f'trpo_mlp{i}',
                                         f'{file_base_trpo}{str(i).zfill(4)}',
                                         MaskableTrpo,
                                         cnn=False,
                                         policy_cls=CustomMlpTrpoPolicy)
                          for i in range(86, 92))

    l1 = list([best_ppo_yet])
    # l2 = list(ppo_base2_cnn())

    for agent2 in range(1):
        bench_both_sides(
            alpha,
            # best_ppo_yet.set_deterministic(False),
            # ga_vpn_5,
            best_ppo_yet,  # .set_deterministic(False),
            # ppo_del2.set_deterministic(False),#agent,
            times=1,
            timed=True,
            verbose=1)
