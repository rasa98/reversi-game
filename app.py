import random
import time

from models.minmax import (mm_static,
                           mm2_dynamic,
                           ga_0,
                           ga_1,
                           ga_2,
                           ga_vpn_5)
from models.ppo_masked_model import (load_model_new,
    # ai385,
    # fixed_330
                                     )
from models.model_interface import ai_random
from models.montecarlo import mcts_model
from models.ParallelMCTS import PMCTS
import models.AlphaZero
from models.AlphaZero import (gen_azero_model,
                              multi_folder_load_some_models)
from bench_agent import bench_both_sides

from game_modes import ai_vs_ai_cli
from collections import Counter
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
                              for i in [17, 18, 19, 20, 21, 22])
    it = multi_ppo_v3v3()
    best_ppo_yet = next(it)
    best_next_18 = next(it)

    file_base_v3v3v1 = 'scripts/rl/output/v3v3-1/history_'
    multi_ppo_v3v3v1 = lambda: (load_model_new(f'ppo_{i}', f'{file_base_v3v3v1}{str(i).zfill(4)}')
                                for i in [24, 32, 33, 37])  # 32 univerzalno bolji

    file_base_v4 = 'scripts/rl/output/v4/history_'
    multi_ppo_v4 = lambda: (load_model_new(f'ppo_{i}', f'{file_base_v4}{str(i).zfill(4)}')
                            for i in [37, 38, 39, 40, 41])

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

    # model_location = f'models/alpha-zero/my_models/vF/model_329.pt'
    # alpha_params = {'hidden_layer': 128, 'res_block': 20}
    # alpha_329 = gen_azero_model(model_location, alpha_params)

    # folder_params = [(f'models/alpha-zero/my_models/v17', alpha_params, range(38, 39))]
    # many_zero_models = multi_folder_load_some_models(folder_params)

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
                                                     f'{file_base}{str(i).zfill(4)}')
                                      for i in [1, 2, 3, 4])

    # l1 = list(multi_dqn())
    for agent in multi_ppo_cnn_paral_v0():
        bench_both_sides(
            ga_vpn_5,
            # best_ppo_yet.set_deterministic(False),
            # ga_vpn_5,
            agent.set_deterministic(False),
            # ppo_del2.set_deterministic(False),#agent,
            times=100,
            timed=True,
            verbose=1)
