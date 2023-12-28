import random

from models.minmax import Minimax, depth_f_default
from heuristics.heu1 import heuristic, heuristic2
from game_modes import ai_vs_ai_cli
from collections import Counter
import timeit
from cProfile import Profile
from pstats import SortKey, Stats

from models.ppo_masked_model import (load_model, load_model_2,
                                     load_model_66, load_model_64_64_1)


def both_sides(ai1, ai2, times=200):
    d1 = benchmark(ai1, ai2, times=times)
    d2 = benchmark(ai2, ai1, times=times)
    d = {ai1['name']: ([1, d1[1]], [2, d2[2]]),
         ai2['name']: ([2, d1[2]], [1, d2[1]]),
         'zeros': ([0, d1.get(0, 0)], [0, d2.get(0, 0)])}
    print(d)
    return d


def benchmark(ai1, ai2, times=200):
    vals = []
    for _ in range(times):
        winner_str = ai_vs_ai_cli(ai1, ai2)
        vals.append(winner_str)
    d_counter = dict(Counter(vals))

    sorted_dict = dict(sorted(d_counter.items(), reverse=True))

    # print(sorted_dict)
    return sorted_dict


def test_models():
    for i in range(200, 201):  # 219, 212, 235, 248, 294 -- 457, 549, --550--, 557, 591, 595, 615

        try:  # 930, 1283, 1319
            f1 = ('training/Dict_obs_space/ppo_masked_selfplay' +
                  '_dict_deterministic_new_rewards/2/history_') + str(i).zfill(8)
            ai1 = {"name": f'ppo_masked_{i}', "first_turn": True, 'f': load_model_66(f1).predict_best_move}

            f2 = ('training/Dict_obs_space/ppo_masked_selfplay' +
                  '_dict_deterministic_new_rewards/2/history_') + str(i + 1).zfill(8)
            ai2 = {"name": f'ppo_masked_{i + 1}', "first_turn": True, 'f': load_model_66(f2).predict_best_move}

            for ai in [ai1, ai2]:
                for aii in [ai2, ai385]:
                    if ai != aii:
                        print(f'{i} - ', end=' ')
                        timeit.timeit(lambda: both_sides(ai, aii,
                                                         times=100),
                                      number=1)
        except FileNotFoundError:
            break


def bench_64_64_1():
    logs = 'training/Dict_obs_space/mppo-1-then-2/log_vs_385.txt'
    for i in range(1, 90000):
        try:
            f = ('training/Dict_obs_space/mppo-1-then-2/history_' +
                 str(i).zfill(8))
            ai_other = {"name": f'ppo_masked_{i}',
                        "first_turn": True,
                        'f': load_model_64_64_1(f).predict_best_move}

            print(f'{i} - ', end=' ')
            d_logs = both_sides(ai385, ai_other, times=200)
            with open(logs, mode='a') as log_file:
                log_file.write(str(d_logs)+'\n')
        except FileNotFoundError:
            break


if __name__ == '__main__':
    mm1 = Minimax(lambda _: 3, heuristic)
    # mm2 = Minimax(lambda _: 4, heuristic2)
    mm2 = Minimax(depth_f_default, heuristic2)

    ai1 = {"name": "Fixed depth=3", "first_turn": True, "f": mm1.predict_best_move}
    ai2 = {"name": "dynamic d", "first_turn": True, "f": mm2.predict_best_move}

    file = '/home/rasa/Desktop/jupyter/rl demo/Othello_try_1/ppo_masked_selfplay/history_00000385.zip'
    ai385 = {"name": 'ppo_masked_385', "first_turn": True, 'f': load_model(file).predict_best_move}

    file2 = 'training/Dict_obs_space/ppo_masked_selfplay_3/history_0433'
    ai_other = {"name": 'ppo_masked_3_433', "first_turn": True, 'f': load_model_2(file2).predict_best_move}

    ai_random = {"name": 'random_model', "first_turn": True, 'f': lambda x: (list(x.valid_moves()), None)}
    # ai_vs_ai_cli(ai1, ai2)

    # execution_time = timeit.timeit(lambda: ai_vs_ai_cli(ai1, ai2), number=200)  # Number of executions
    # execution_time = timeit.timeit(lambda: benchmark(ai_random, ai3, times=1000), number=1)
    # execution_time += timeit.timeit(lambda: benchmark(ai3, ai_random, times=1000), number=1)
    # execution_time += timeit.timeit(lambda: benchmark(ai1, ai3, times=1000), number=1)
    # execution_time += timeit.timeit(lambda: benchmark(ai3, ai1, times=1000), number=1)
    # execution_time += timeit.timeit(lambda: benchmark(ai2, ai3, times=1000), number=1)
    # execution_time += timeit.timeit(lambda: benchmark(ai3, ai2, times=1000), number=1)

    # ms = [136, 189, 205, 277, 288, 306, 314, 321, 323, 327, 328, 330, 365,
    #       372, 444, 450, 453, 455, 457, 469, 472, 475, 505, 558, 559, 955,
    #       998, 1046, 1171, 1190, 1191, 1196, 1198, 1209, 1210, 1221, 1387,
    #       1512, 1630] # best -> 327, 450, 455, 469, 1198,
    #
    # for i in ms:  # 219, 212, 235, 248, 294 -- 457, 549, --550--, 557, 591, 595, 615
    #
    #     try:  # 136, 189, 205 new_rewards/2fixed, 277, 288, 306, 314, 321, 323, 327, 328, 330, 365
    #         # 372, 444, 450, 453, 455, 457, 469, 472, 475, 505, 558, 559, 955, 998, 1046, 1171, 1190/1, 1196/8
    #         # 1209/10, 1221, 1387, 1512, 1630
    #         file2 = 'training/Dict_obs_space/ppo_masked_selfplay_dict_deterministic_new_rewards/2fixed/history_' + str(
    #             i).zfill(8)
    #         ai_other = {"name": f'ppo_masked_{i}', "first_turn": True, 'f': load_model_66(file2).predict_best_move}
    #
    #         print(f'{i} - ', end=' ')
    #         execution_time = timeit.timeit(lambda: both_sides(ai385, ai_other,
    #                                                           times=200),
    #                                        number=1)
    #     except FileNotFoundError:
    #         break

    bench_64_64_1()

    # test_models()

    # execution_time = timeit.timeit(lambda: benchmark(ai385, ai_other, times=100), number=1)  # Number of executions
    # execution_time += timeit.timeit(lambda: benchmark(ai_other, ai385, times=100), number=1)
    # print(f"\nExecution time: {execution_time:.6f} seconds")

    # with Profile() as profile:
    #
    #     print(f"{benchmark(ai1, ai2, times=100)}")
    #     (
    #         Stats(profile).
    #         strip_dirs().
    #         sort_stats(SortKey.CALLS).
    #         print_stats()
    #     )
