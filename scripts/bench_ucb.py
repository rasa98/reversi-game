import numpy as np
import numba, math
from numba import types, njit

import timeit
import struct


def get_uct(parent_visits, vis, val):
    avg_val = val / vis
    c = 1.42
    exploration_term = c * math.sqrt(math.log(parent_visits) / vis)
    return avg_val + exploration_term


@njit
def get_uct_fast_sqrt(parent_visits, vis, val):
    avg_val = val / vis
    c = 1.42
    exploration_term = c * fast_inverse_sqrt(math.log(parent_visits) / vis)
    return avg_val + exploration_term


# @njit
# def get_uct_numpy(parent_visits, vis, val):
#     avg_val = val / vis
#     c = 1.42
#     exploration_term = c * np.sqrt(np.log(parent_visits) / vis)
#     return avg_val + exploration_term


parent = 197
vis = 37
val = 76

uct_time = timeit.timeit(lambda: get_uct(parent, vis, val), number=100_000_00)
np_uct_time = timeit.timeit(lambda: get_uct_fast_sqrt(parent, vis, val), number=100_000_00)

print(f"uct: {uct_time}")
print(f"fast inverse uct: {np_uct_time}")
