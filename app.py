import random
import time

from models.minmax import (mm_static,
                           mm2_dynamic,
                           ga_0,
                           ga_1,
                           ga_2,
                           ga_vpn_5)
from models.ppo_masked_model import (ai385,
                                     fixed_330)
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


def benchmark(ai1, ai2, times=200):
    vals = []
    for _ in range(times):
        winner_str = ai_vs_ai_cli(ai1, ai2)
        vals.append(winner_str)
    counter = Counter(vals)
    d = {}
    d[f"{ai1}"] = counter[1]
    d[f"{ai2}"] = counter[2]
    d[f"draw"] = counter[0]

    print(f'{"-" * 5} {ai1} vs {ai2} {"-" * 5}')
    print(d, "\n")
    return counter


@time_function
def bench_both_sides(ai1, ai2, times=200):
    c1 = benchmark(ai1, ai2, times=times)
    c2 = benchmark(ai2, ai1, times=times)
    return dict(c1 + c2)


if __name__ == '__main__':
    pmcts = PMCTS('parallel mcts', time_limit=5.25, iter_limit=6250)

    with pmcts.create_pool_manager(num_processes=4):
        bench_both_sides(ai385,
                         pmcts,
                         # mcts_model,
                         times=10)
