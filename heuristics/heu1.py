import numpy as np

if __name__ != '__main__':
    from game_logic import Othello
    from heuristics.critical_fields import danger_fields, safe_fields, corners
else:
    from critical_fields import danger_fields, safe_fields, corners
    Othello = None


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
    turn = white_count + black_count - 3  # 60 turns to play
    factor = f(turn)
    return factor * (white_count - black_count)


def count_corners(board: np.ndarray, f=lambda x: ((60 - x) // 3) ** 2):
    # who has more => better
    c = corners(board)
    white_corners, black_corners = count_white_black(c)
    white, black = count_white_black(board)
    turn = white + black - 3  # 60 turns to play
    factor = f(turn)
    return factor * (white_corners - black_corners)


def count_danger_early_game(board: np.ndarray, f=lambda x: (65 - x) // 10):
    if np.any(safe_fields(board) == 0):  # if board still has moves in center, then penalize
        danger_view = danger_fields(board)
        white, black = count_white_black(danger_view)
        turn = white + black - 3  # 60 turns to play
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
           # count_corners(board) +
           count_danger_early_game(board))
    return res


def heuristic2(game: Othello):
    board = game.board
    res = (count_chips(board, lambda turn: (turn // 5) + 1) +
           count_corners(board) +
           count_danger_early_game(board, lambda turn: (65 - turn) // 5))
    return res


if __name__ == '__main__':
    board = [[1, 1, 1, 1, 1, 0, 0, 1],
             [2, 1, 0, 2, 2, 0, 0, 2],
             [0, 2, 1, 0, 0, 0, 2, 2],
             [2, 2, 0, 0, 0, 0, 0, 2],
             [0, 0, 1, 1, 1, 0, 2, 2],
             [2, 1, 0, 2, 1, 2, 1, 0],
             [0, 0, 0, 1, 1, 2, 1, 2],
             [1, 2, 2, 2, 2, 0, 2, 1]]
    board = np.array(board)
    c = count_corners(board)
    print(c)