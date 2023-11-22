import numpy as np

from game_logic import Othello
from heuristics.critical_fields import danger_fields, safe_fields, corners


def count_white_black(board: np.ndarray):
    white = np.count_nonzero(board == 1)
    black = np.count_nonzero(board == 2)
    return white, black


def count_chips(board: np.ndarray, f=lambda x: (x // 10) + 1):
    # White is maximizing, black is minimizing
    # So if white has more chips it will be positive.
    # The more it has the position is "better" if its near end,
    # but everything can change in a few moves
    white_count, black_count = count_white_black(board)
    turn = white_count + black_count - 4  # 60 turns to play
    factor = f(turn)
    return factor * (white_count - black_count)


def count_borders(board: np.ndarray, factor):
    # who has more => better
    c = corners(board)
    white, black = count_white_black(c)
    return factor * (white - black)


def count_danger_early_game(board: np.ndarray, f=lambda x: (65 - x) // 10):
    if np.any(safe_fields(board) == 0):  # if board still has moves in center, then penalize
        danger_view = danger_fields(board)
        white, black = count_white_black(danger_view)
        turn = white + black - 4  # 60 turns to play
        factor = f(turn)
        return factor * (black - white)  # if black has more it will be positive => bad for black (minimizing)
    return 0


def count_safer(board: np.ndarray, ):
    safe_view = safe_fields(board)
    white, black = count_white_black(safe_view)


#   todo finish this method


def heuristic(game: Othello):
    board = game.board
    res = (count_chips(board) +
           count_borders(board, 50) +
           count_danger_early_game(board))
    return res


def heuristic2(game: Othello):
    board = game.board
    res = (count_chips(board, lambda x: (x // 5) + 1) +
           count_borders(board, 10) +
           count_danger_early_game(board, lambda x: (65 - x) // 5))
    return res
