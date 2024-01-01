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


def count_danger_early_game(stats, f=lambda x: (65 - x) // 10):
    _, turn, board = stats
    if np.any(safe_fields(board) == 0):  # if board still has moves in center, then penalize
        danger_view = danger_fields(board)
        white, black = count_white_black(danger_view)

        factor = f(turn)
        return factor * (black - white)  # if black has more it will be positive => bad for black (minimizing)
    return 0


def count_safer(stats, f=lambda x: (65 - x) // 10):
    _, turn, board = stats
    safe_view = safe_fields(board)
    white, black = count_white_black(safe_view)

    factor = f(turn)
    return factor * (white - black)


def minimize_opponent_moves(game: Othello, max_score):
    factor = -1 if game.last_turn == 2 else 1  # if minimizer -> -1, maximizer -> 1
    opponent_num_of_moves = 0
    if game.player_turn != game.last_turn:  # if they are equal, means opponent didnt have any move.
        opponent_num_of_moves = len(game.valid_moves())
    score = max_score * (1 / ((opponent_num_of_moves + 1) ** 1.5))
    return factor * score


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
           count_danger_early_game(stats, lambda turn: (65 - turn) // 5))
    return res


def create_heuristic(chip_divisor, corner_divisor, corner_exponent, danger_divisor):
    # chip_divisor      - [1, 2, 3..., 20]
    # corner_divisor    - [1, 2, 3..., 20]
    # corner_exponent   - range [1.0 - 3.0]
    # danger_divisor    - [1, 2, 3..., 20]
    def heu(game: Othello):
        stats = (game.chips, game.turn, game.board)
        res = (count_chips(stats, lambda turn: (turn // chip_divisor) + 1) +
               count_corners(stats, lambda x: ((60 - x) // corner_divisor) ** corner_exponent) +
               count_danger_early_game(stats, lambda turn: (65 - turn) // danger_divisor))
        return res

    return heu


def create_heuristic2(chip_divisor, corner_divisor, corner_exponent, danger_divisor):
    # chip_divisor      - [1, 2, 3..., 20]
    # corner_divisor    - [1, 2, 3..., 20]
    # corner_exponent   - range [1.0 - 3.0]
    # danger_divisor    - [1, 2, 3..., 20]
    def heu(game: Othello):
        stats = (game.chips, game.turn, game.board)
        res = (count_chips(stats, lambda turn: (turn // chip_divisor) + 1) +
               count_corners(stats, lambda x: ((60 - x) // corner_divisor) ** corner_exponent) +
               count_danger_early_game(stats, lambda turn: (65 - turn) // danger_divisor))
        return res

    return heu


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
    c = count_corners((_, 35, board))
    print(c)
