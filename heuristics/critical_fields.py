import numpy as np
from numba import njit


def corners(board: np.ndarray):
    return board[[0, 0, -1, -1], [0, -1, 0, -1]]


def safe_fields(board: np.ndarray):
    return board[[2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5],
                 [2, 3, 4, 5, 2, 5, 2, 5, 2, 3, 4, 5]]


def danger_fields(board: np.ndarray):
    return board[[0, 0, 1, 1, 1, 1, -2, -2, -2, -2, -1, -1],
                 [1, -2, 0, 1, -2, -1, 0, 1, -2, -1, 1, -2]]


@njit(cache=True)
def corners2(board: np.ndarray):
    res = []
    res.append(board[0, 0])
    res.append(board[0, -1])
    res.append(board[-1, 0])
    res.append(board[-1, -1])
    return np.array(res)


# @njit(cache=True)
# def safe_fields2(board: np.ndarray):
#     return board[[2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5],
#                  [2, 3, 4, 5, 2, 5, 2, 5, 2, 3, 4, 5]]
#
#
# @njit(cache=True)
# def danger_fields2(board: np.ndarray):
#     return board[[0, 0, 1, 1, 1, 1, -2, -2, -2, -2, -1, -1],
#                  [1, -2, 0, 1, -2, -1, 0, 1, -2, -1, 1, -2]]

if __name__ == "__main__":
    import time

    board = [[1, 1, 1, 1, 1, 0, 0, 1],
             [2, 1, 0, 2, 2, 0, 0, 2],
             [0, 2, 1, 0, 0, 0, 2, 2],
             [2, 2, 0, 0, 0, 0, 0, 2],
             [0, 0, 1, 1, 1, 0, 2, 2],
             [2, 1, 0, 2, 1, 2, 1, 0],
             [0, 0, 0, 1, 1, 2, 1, 2],
             [1, 2, 2, 2, 2, 0, 2, 2]]
    board = np.array(board)

    times = 150000
    start = time.perf_counter()
    for i in range(times):
        a = corners(board)
    end = time.perf_counter()
    print(f'time needed for counting corners: {end - start} secs')

    # ------------------

    start = time.perf_counter()
    for i in range(times):
        b = corners2(board)
    end = time.perf_counter()
    print(f'time needed for counting corners2: {end - start} secs')

    assert a.tolist() == b.tolist()