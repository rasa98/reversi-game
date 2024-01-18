import random
import time

from models.minmax import Minimax, depth_f_default
from heuristics.heu1 import heuristic, heuristic2
from heuristics.heu2 import create_heuristic

from heuristics.ga.heu_ga import create_heuristic as create_heuristic2

from game_modes import ai_vs_ai_cli
from collections import Counter
import timeit
from cProfile import Profile
from pstats import SortKey, Stats

from models.ppo_masked_model import (load_model, load_model_2,
                                     load_model_66, load_model_64_64_1,
                                     load_model_64_2_1)


def both_sides(ai1, ai2, times=200):
    d1 = benchmark(ai1, ai2, times=times)
    d2 = benchmark(ai2, ai1, times=times)
    d = {ai1['name']: ([1, d1.get(1, 0)], [2, d2.get(2, 0)]),
         ai2['name']: ([2, d1.get(2, 0)], [1, d2.get(1, 0)]),
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
                log_file.write(str(d_logs) + '\n')
        except FileNotFoundError:
            break


def bench_64_2_1():
    model_path = 'training/Dict_obs_space/mppo_num_chips/models/history_00000414'

    model = {"name": f'ppo_masked_414',
             "first_turn": True,
             'f': load_model_64_2_1(model_path).predict_best_move}

    both_sides(model, ai_0, times=100)


if __name__ == '__main__':
    mm1 = Minimax(lambda _: 1, heuristic)
    # mm2 = Minimax(lambda _: 4, heuristic2)

    # depth_f = depth_f_default
    depth_f = lambda _: 1
    mm2 = Minimax(depth_f, heuristic2)

    ai1 = {"name": "Fixed depth=3", "first_turn": True, "f": mm1.predict_best_move}
    ai2 = {"name": "dynamic d", "first_turn": True, "f": mm2.predict_best_move}

    file = '/home/rasa/Desktop/jupyter/rl demo/Othello_try_1/ppo_masked_selfplay/history_00000385.zip'
    ai385 = {"name": 'ppo_masked_385', 'f': load_model(file).predict_best_move}

    file = 'training/Dict_obs_space/mppo_num_chips/models/history_00000330'
    fixed_330 = {"name": 'fixed_ppo_masked_330', 'f': load_model_64_2_1(file).predict_best_move}

    # file2 = 'training/Dict_obs_space/mppo-1-then-2/history_' + str(354).zfill(8)
    # ai_other = {"name": 'ppo_masked_64_64_1_354', 'f': load_model_64_64_1(file2).predict_best_move}

    ai_random = {"name": 'random_model', "first_turn": True, 'f': lambda x: (list(x.valid_moves()), None)}

    # heu_params = 19.70822410606773, 1.3048221582220656, 2.9128158081863784, 1.351857641596074
    # heu_params = 19.71396080843101, 1.1239482682808002, 2.4049454912356936, 1.1732313053633865
    heu_params = 14.110659370384004, 1.6725387766365891, 2.7262102930984993, 1.0988509935146311, 946.9075924700552

    mm_no_depth = Minimax(lambda _: 1, create_heuristic(*heu_params))
    ai_0 = {"name": "depth 1 GA", "first_turn": True, "f": mm_no_depth.predict_best_move}

    # corner divisor: 1.141624324278253, corner exponent: 2.2005380364531657,
    # danger divisor: 1.5537486913611744,
    # min opp score: 202.78118735939893
    from heuristics.ga.heu_func import (CountChips, CountDangerEarlyGame, CountCorners,
                                        MinimizeOpponentMoves)

    depth_f_default = lambda _: 1
    custom_heu = {CountCorners: CountCorners(1.141624324278253, 2.2005380364531657),
                  CountDangerEarlyGame: CountDangerEarlyGame(1.5537486913611744),
                  MinimizeOpponentMoves: MinimizeOpponentMoves(202.78118735939893)}
    ga_custom_heu = {"name": "depth 1 GA-custom heu",
                     "first_turn": True,
                     "f": Minimax(depth_f_default, create_heuristic2(custom_heu)).predict_best_move}

    custom_heu_2 = {CountCorners: CountCorners(1.5186058726512501, 2.5149087893017823),
                    CountDangerEarlyGame: CountDangerEarlyGame(6.374267148895484),
                    MinimizeOpponentMoves: MinimizeOpponentMoves(215.8041817188843)}
    ga_custom_2 = {"name": "depth 1 GA-custom-2 heu",
                   "first_turn": True,
                   "f": Minimax(depth_f_default, create_heuristic2(custom_heu_2)).predict_best_move}

    start = time.perf_counter()
    both_sides(fixed_330, ai385, times=1000)
    end = time.perf_counter()
    print(f'time needed {end - start}')

    # bench_64_2_1()

    # time_amount = timeit.timeit(lambda: both_sides(ai_0, ai_random, times=100), number=1)
    # print(f'time needed {time_amount}')

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

    # test_models()

    # execution_time = timeit.timeit(lambda: benchmark(ai385, ai_other, times=100), number=1)  # Number of executions
    # execution_time += timeit.timeit(lambda: benchmark(ai_other, ai385, times=100), number=1)
    # print(f"\nExecution time: {execution_time:.6f} seconds")

    # with Profile() as profile:
    #
    #     print(f"{both_sides(ai385, ga_custom_2, times=100)}")
    #     (
    #         Stats(profile).
    #         strip_dirs().
    #         sort_stats(SortKey.CALLS).
    #         print_stats()
    #     )
