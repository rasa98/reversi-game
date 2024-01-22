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
    d_counter = dict(Counter(vals))

    sorted_dict = dict(sorted(d_counter.items(), reverse=True))

    # print(sorted_dict)
    return sorted_dict


@time_function
def both_sides(ai1, ai2, times=200):
    d1 = benchmark(ai1, ai2, times=times)
    d2 = benchmark(ai2, ai1, times=times)
    d = {str(ai1): ([1, d1.get(1, 0)], [2, d2.get(2, 0)]),
         str(ai2): ([2, d1.get(2, 0)], [1, d2.get(1, 0)]),
         'zeros': ([0, d1.get(0, 0)], [0, d2.get(0, 0)])}
    print(d)
    return d


if __name__ == '__main__':
    both_sides(fixed_330, ga_vpn_5, times=100)

    # profile(lambda: print(both_sides(ga_custom_2, ai_random, times=10)))
