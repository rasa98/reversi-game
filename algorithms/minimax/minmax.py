import numpy as np
from game_logic import Othello


def fixed_depth_f(d):
    return lambda _: d


def dynamic_depth_f(turn):
    if turn <= 5:
        return int(-0.7 * (turn - 4) + 4)
    elif turn <= 45:
        return 3
    elif turn <= 50:
        return int(0.7 * (turn - 45) + 3)
    else:
        return 10


class Minimax:
    def __init__(self, depth_f, heu):
        self.depth = None
        self.depth_f = depth_f
        # self.maximizing_player = maximizing_player
        self.heu = heu
        self.best_estimate = None
        self.best_moves_to_play = []
        self.all_moves = []
        self.maximizer = None

    def best_moves(self):
        assert len(self.all_moves) != 0, "need first to simulate move!!!"
        return self.best_moves_to_play

    def simulate(self, game):
        self.best_moves_to_play = []
        self.all_moves = []
        self.maximizer = True if game.player_turn == 1 else False  # player 1 maximizes

        self.depth = self.depth_f(game.turn)  # depth >= 1
        self.best_estimate = self._main(self.depth,
                                        float('-inf'),  # alpha
                                        float('inf'),  # beta
                                        game.get_snapshot())

        arr = np.array([val for _, val in self.all_moves])
        if self.maximizer is False:  # if we are minimizing (playing as player 2) invert the vals
            arr *= -1
        probs = scaled_softmax(arr)

        action_probs = np.zeros(game.action_space())
        for idx, (move, eval) in enumerate(self.all_moves):
            if eval == self.best_estimate:
                self.best_moves_to_play.append(move)

            encoded_move = game.__class__.get_encoded_field(move)
            action_probs[encoded_move] = probs[idx]
        # action_probs /= np.sum(action_probs)

        return action_probs

    def _main(self, depth, a, b, game: Othello):
        winner = game.get_winner()
        if winner is None:
            pass
        elif winner == 1:
            return 100_000
        elif winner == 2:
            return -100_000
        else:  # when winner is 0 , meaning its draw
            return 0

        if depth == 0:
            return self.heu(game)

        if game.player_turn == 1:  # Player 1 maximizes
            return self.maximize(depth, a, b, game)
        else:
            return self.minimize(depth, a, b, game)

    def maximize(self, depth, a, b, game: Othello):
        max_eval = float('-inf')
        for field in game.valid_moves_sorted():
            game_copy = game.get_snapshot()
            game_copy.play_move(field)

            eval = self._main(depth - 1, a, b, game_copy)

            max_eval = max(max_eval, eval)
            a = max(a, eval)

            if depth == self.depth:
                self.all_moves.append((field, eval))

            if b <= a:
                break

        return max_eval

    def minimize(self, depth, a, b, game):
        min_eval = float('inf')
        for field in game.valid_moves_sorted():
            game_copy = game.get_snapshot()
            game_copy.play_move(field)

            eval = self._main(depth - 1, a, b, game_copy)

            min_eval = min(min_eval, eval)
            b = min(b, eval)

            if depth == self.depth:
                self.all_moves.append((field, eval))

            if b <= a:
                break

        return min_eval


def scaled_softmax(values):
    # Check if a high win value is present
    has_win = np.any(values >= 90000)
    
    # Determine the temperature based on the presence of a high win value
    temp = 0.05 if has_win else 0.4
    
    # Adjust values to handle extreme negative values
    ignore_mask = (values == -100000)
    valid_values = values[~ignore_mask]

    if len(valid_values) == 0:
        # If all values are extremely negative, assign probability 1.0 to those values
        probabilities = np.zeros_like(values, dtype=float)
        probabilities[ignore_mask] = 1.0
        return probabilities / np.sum(probabilities)

    # If there's only one valid value, assign probability 1.0 to that value
    if len(valid_values) == 1:
        probabilities = np.zeros_like(values, dtype=float)
        probabilities[values == valid_values[0]] = 1.0
        return probabilities

    # Min-max normalization excluding extreme negative values
    max_val = np.max(valid_values)
    min_val = np.min(valid_values)

    if min_val == max_val:
        # If min_val equals max_val, all valid values are the same
        scaled_values = np.zeros_like(values, dtype=float)
        scaled_values[~ignore_mask] = 1.0
    else:
        scaled_values = np.zeros_like(values, dtype=float)
        scaled_values[~ignore_mask] = (valid_values - min_val) / (max_val - min_val)

    # Apply scaled softmax with temperature
    exp_values = np.exp(scaled_values / temp)
    exp_values[ignore_mask] = 0  # Ensure -100000 values have zero probability
    probabilities = exp_values / np.sum(exp_values)

    return probabilities
