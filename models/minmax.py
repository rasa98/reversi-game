from game_logic import Othello


class Minimax:
    def __init__(self, depth, game, heu):
        self.depth = depth
        self.game = game
        # self.maximizing_player = maximizing_player
        self.heu = heu

        self.best_fields = []
        # self.all_moves = []
        self.best_estimate = self._main(depth, float('-inf'), float('inf'), game)

    def get_fields_and_estimate(self):
        return self.best_fields, self.best_estimate

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

    def maximize(self, depth, a, b, game):
        max_eval = float('-inf')

        for field in game.valid_moves():
            next_game_position = game.get_snapshot().play_move(field)
            eval = self._main(depth - 1, a, b, next_game_position)
            new_max_eval = max(max_eval, eval)
            a = max(a, eval)

            if depth == self.depth:
                if eval == new_max_eval and eval == max_eval:
                    self.best_fields.append(field)
                elif max_eval < new_max_eval:
                    self.best_fields = [field]
                else:
                    pass

            if b <= a:
                break
            max_eval = new_max_eval
        return max_eval

    def minimize(self, depth, a, b, game):
        min_eval = float('inf')

        for field in game.valid_moves():
            next_game_position = game.get_snapshot().play_move(field)
            eval = self._main(depth - 1, a, b, next_game_position)
            new_min_eval = min(min_eval, eval)
            b = min(b, eval)

            if depth == self.depth:
                if eval == new_min_eval and eval == min_eval:
                    self.best_fields.append(field)
                elif min_eval > new_min_eval:
                    self.best_fields = [field]
                else:
                    pass

            if b <= a:
                break
            min_eval = new_min_eval
        return min_eval
