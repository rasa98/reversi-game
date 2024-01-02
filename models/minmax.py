import numpy as np
from models.model_interface import ModelInterface
from game_logic import Othello


def depth_f_default(turn):
    if turn <= 5:
        return int(-0.7 * (turn - 4) + 4)
    elif turn <= 45:
        return 3
    elif turn <= 50:
        return int(0.7 * (turn - 45) + 3)
    else:
        return 10


class Minimax(ModelInterface):
    def __init__(self, depth_f, heu):
        self.depth = None
        self.depth_f = depth_f
        # self.maximizing_player = maximizing_player
        self.heu = heu
        self.best_fields = []
        self.all_moves = []

    def predict_best_move(self, game):
        self.best_fields.clear()
        self.all_moves.clear()

        self.depth = self.depth_f(game.turn)  # depth >= 1
        best_estimate = self._main(self.depth, float('-inf'), float('inf'), game.get_snapshot())
        # print(f'turn: {turn}, --- choose depth: {self.depth} --- estimate: {best_estimate}, All moves: {self.all_moves}')

        # print(f'All moves: {self.all_moves}')
        return tuple(self.best_fields), best_estimate

    def _main(self, depth, a, b, game: Othello):
        match game.get_winner():
            case 0:
                return 0
            case 1:
                return 100000
            case 2:
                return -100000
            case _:  # if None has yet won
                pass
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
            biggest_max_eval_yet = max(max_eval, eval)
            a = max(a, eval)

            if depth == self.depth:
                self.all_moves.append((field, eval))
                if eval == biggest_max_eval_yet and eval == max_eval:
                    self.best_fields.append(field)
                elif max_eval < eval:
                    self.best_fields = [field]
                else:
                    pass
            max_eval = biggest_max_eval_yet
            if b <= a:
                break

        return max_eval

    def minimize(self, depth, a, b, game):
        min_eval = float('inf')
        for field in game.valid_moves_sorted():
            game_copy = game.get_snapshot()
            game_copy.play_move(field)

            eval = self._main(depth - 1, a, b, game_copy)
            try:
                new_min_eval = min(min_eval, eval)
            except TypeError:
                print(f'min_eval = {min_eval}, eval = {eval}')
                raise TypeError()
            b = min(b, eval)

            if depth == self.depth:
                self.all_moves.append((field, eval))
                if eval == new_min_eval and eval == min_eval:
                    self.best_fields.append(field)
                elif min_eval > eval:
                    self.best_fields = [field]
                else:
                    pass
            min_eval = new_min_eval

            if b <= a:
                break

        return min_eval


if __name__ == '__main__':
    print([depth_f_default(x) for x in range(61)])
