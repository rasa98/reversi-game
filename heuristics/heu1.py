import numpy as np
from numba import njit

if __name__ == '__main__':
    import os
    import sys

    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
    sys.path.append(source_dir)

from game_logic import Othello
from heuristics.critical_fields import danger_fields, safe_fields, corners, corners2


@njit(cache=True)
def count_white_black(board: np.ndarray):
    white = 0
    black = 0

    for element in board.reshape(-1):
        if element == 1:
            white += 1
        elif element == 2:
            black += 1

    return white, black



def count_chips(stats, f=lambda x: (x // 10) + 1):
    # White is maximizing, black is minimizing
    # So if white has more chips it will be positive.
    # The more it has the position is "better" if its near end,
    # but everything can change in a few moves
    ((white_count, black_count), turn, _) = stats

    factor = f(turn)
    return factor * (white_count - black_count)


def count_corners(stats, f=lambda x: ((60 - x) // 3) ** 2):
    # who has more => better
    _, turn, board = stats
    c = corners(board)
    white_corners, black_corners = count_white_black(c)
    factor = f(turn)
    return factor * (white_corners - black_corners)


def count_danger_early_game(stats, f=lambda x: (25 - x) // 5):
    _, turn, board = stats
    if turn <= 20:  # np.any(safe_fields(board) == 0):  # if board still has moves in center, then penalize
        danger_view = danger_fields(board)
        white, black = count_white_black(danger_view)

        factor = f(turn)
        return factor * (black - white)  # if black has more it will be positive => bad for black (minimizing)
    return 0


def count_safer(stats, f=lambda x: 1):
    _, turn, board = stats
    if turn <= 20:
        safe_view = safe_fields(board)
        white, black = count_white_black(safe_view)

        factor = f(turn)
        return factor * (white - black)
    return 0


def max_my_moves(game: Othello, max_score):
    """
    if opponent didnt have a move returns 3 x max_score.
    else the more moves you have available the more score it returns
    """
    sign = 1 if game.player_turn == 1 else -1  # if minimizer -> -1, maximizer -> 1
    if game.player_turn != game.last_turn:  # if they are equal, means opponent didnt have any move.
        my_num_of_moves = len(game.valid_moves())
        new_score = max_score * (1 - (1 / (my_num_of_moves * 1.25)))
        return sign * new_score
    else:
        return sign * 3 * max_score


def heuristic(game: Othello):
    stats = (game.chips, game.turn, game.board)
    res = (count_chips(stats) +
           # count_corners(stats, board) +
           count_danger_early_game(stats))
    return res


def heuristic2(game: Othello):
    stats = (game.chips, game.turn, game.board)
    res = (count_chips(stats, lambda turn: (turn // 5) + 1) +
           count_corners(stats) +
           count_danger_early_game(stats, lambda turn: (25 - turn) // 3))
    return res


if __name__ == '__main__':
    import time

    board = [[1, 1, 1, 1, 1, 0, 0, 1],
             [2, 1, 0, 2, 2, 0, 0, 2],
             [0, 2, 1, 0, 0, 0, 2, 2],
             [2, 2, 0, 0, 0, 0, 0, 2],
             [0, 0, 1, 1, 1, 0, 2, 2],
             [2, 1, 0, 2, 1, 2, 1, 0],
             [0, 0, 0, 1, 1, 2, 1, 2],
             [1, 2, 2, 2, 2, 0, 2, 1]]
    board = np.array(board)

    c = count_corners(("", 35, board))

    print(c)

    mat_gen_f = lambda: np.random.randint(0, 3, size=(8, 8))


    times = 10000
    mats = [mat_gen_f() for _ in range(times)]

    start = time.perf_counter()

    for i in range(times):
        count_white_black(mats[i])

    end = time.perf_counter()
    print(f'time needed: {end - start} secs')

