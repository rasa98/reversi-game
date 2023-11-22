import numpy as np


def corners(board: np.ndarray):
    return board[[0, 0, -1, -1], [0, -1, 0, -1]]


def safe_fields(board: np.ndarray):
    return board[2:6, 2:6]  # todo: exlude center positions


def danger_fields(board: np.ndarray):
    return board[[0, 0, 1, 1, 1, 1, -2, -2, -2, -2, -1, -1],
    [1, -2, 0, 1, -2, -1, 0, 1, -2, -1, 1, -2]]
