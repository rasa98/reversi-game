import numpy as np


def corners(board: np.ndarray):
    return board[[0, 0, -1, -1], [0, -1, 0, -1]]


def safe_fields(board: np.ndarray):
    return board[[2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5],
                 [2, 3, 4, 5, 2, 5, 2, 5, 2, 3, 4, 5]]


def danger_fields(board: np.ndarray):
    return board[[0, 0, 1, 1, 1, 1, -2, -2, -2, -2, -1, -1],
                 [1, -2, 0, 1, -2, -1, 0, 1, -2, -1, 1, -2]]
