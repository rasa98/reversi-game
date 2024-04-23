import random
import time

from models.minmax import (mm_static,
                           mm2_dynamic,
                           ga_0,
                           ga_1,
                           ga_2,
                           ga_vpn_5)
# from models.ppo_masked_model import (ai385,
#                                      fixed_330)
from models.model_interface import ai_random
from models.montecarlo import mcts_model
from models.ParallelMCTS import PMCTS

from game_modes import ai_vs_ai_cli
from collections import Counter
from cProfile import Profile
from pstats import SortKey, Stats


def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Time needed: {end - start} seconds')
        return res

    return wrapper


def profile(lambda_my_code):
    with Profile() as profiler:
        lambda_my_code()

    # Print or manipulate the profiling stats as needed
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CALLS)
    stats.print_stats()


def print_results(player1, player2, d):
    print(f'{"-" * 5} {player1} vs {player2} {"-" * 5}')
    print(d, "\n")


def benchmark(ai1, ai2, times=200, verbose=True):
    vals = []
    for _ in range(times):
        winner_str = ai_vs_ai_cli(ai1, ai2)
        vals.append(winner_str)
    counter = Counter(vals)
    d = {}
    d[f"{ai1}"] = counter[1]
    d[f"{ai2}"] = counter[2]
    d[f"draw"] = counter[0]

    if verbose:
        print_results(ai1, ai2, d)
    return d


@time_function
def bench_both_sides(ai1, ai2, times=200):
    d1 = benchmark(ai1, ai2, times=times, verbose=False)
    d2 = benchmark(ai2, ai1, times=times, verbose=False)

    print_results(ai1, ai2, d1)
    print_results(ai2, ai1, d2)


if __name__ == '__main__':
    pmcts = PMCTS('parallel mcts',
                  time_limit=1,
                  iter_limit=1000
                  )

    with PMCTS.create_pool_manager(pmcts, num_processes=4):
        bench_both_sides(ai_random,
                         pmcts,
                         # mcts_model,
                         times=1)
