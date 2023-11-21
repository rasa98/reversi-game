import numpy as np

directions = {(-1, -1), (-1, 0), (-1, 1),
              (0, -1), (0, 1),
              (1, -1), (1, 0), (1, 1)}


class Othello:
    def __init__(self, players=("w", "b"), board=None, first_move=1):
        self.white, self.black = players
        self.player_turn = first_move
        self.valid_moves_to_reverse = None
        self.winner = None
        if board:
            self.board = np.array(board)
        else:
            self._set_default_board()
        self._calculate_next_valid_moves()
        self._check_correctness()

    def get_snapshot(self):
        return Othello(players=(self.white, self.black), board=np.copy(self.board), first_move=self.player_turn)

    def _swap_player_turn(self):
        self.player_turn = 3 - self.player_turn

    def _set_default_board(self):
        self.board = np.full((8, 8), 0)
        self.board[3:5, 3:5] = [[1, 2],
                                [2, 1]]

    def valid_moves(self):
        return set(self.valid_moves_to_reverse.keys())

    def _calculate_next_valid_moves(self):
        moves_to_reverse = {}
        for field in self._get_edge_fields():
            if to_reverse := self._get_reversed_fields(field):
                moves_to_reverse[field] = to_reverse
        self.valid_moves_to_reverse = moves_to_reverse

    def _get_reversed_fields(self, field):
        s = set()
        maybe_s = set()
        row, col = field

        for x, y in directions:
            for times in range(1, 8):
                new_row = row + x * times
                new_col = col + y * times

                try:
                    match self.board[new_row, new_col]:
                        case 0:
                            break
                        case self.player_turn:
                            if times == 1:
                                break
                            else:
                                s = s.union(maybe_s)
                        case _:  # opponent case
                            maybe_s.add((new_row, new_col))
                except IndexError:
                    break
            maybe_s.clear()

        if not len(s):
            return None
        return s

    def _get_edge_fields(self):
        player_positions = np.where(self.board == 3 - self.player_turn)  # Get positions of the specified player's moves

        empty_adjacent = set()

        for i, j in zip(player_positions[0], player_positions[1]):
            neighbor_slice = self.board[max(0, i - 1):min(8, i + 2), max(0, j - 1):min(8, j + 2)]
            empty_spots = np.argwhere(neighbor_slice == 0) + np.array([max(0, i - 1), max(0, j - 1)])

            for spot in empty_spots:
                x, y = spot
                if np.any(self.board[max(0, x - 1):min(8, x + 2), max(0, y - 1):min(8, y + 2)] != 0):
                    empty_adjacent.add(tuple(spot))

        return empty_adjacent

    def play_move(self, field):
        if not self.winner:  # if the game haven't ended, you can play
            swap_value = self.player_turn

            if field in self.valid_moves():
                to_reverse = list(self.valid_moves_to_reverse[field])
                to_reverse.append(field)
                xs = np.array([x[0] for x in to_reverse], dtype=int)
                ys = np.array([x[1] for x in to_reverse], dtype=int)

                self.board[xs, ys] = np.array([swap_value] * len(to_reverse), dtype=int)
                self._update_state()
                self._check_correctness()

    def _update_state(self):
        self._swap_player_turn()
        self._calculate_next_valid_moves()

    def _check_correctness(self):
        if not self.valid_moves_to_reverse:
            self._update_state()
            if not self.valid_moves_to_reverse:
                winner, amount, loser_amount = self.count_winner()
                if winner == "draw":
                    print(f"Its a draw - {amount} : {loser_amount}")
                else:
                    print(f"{winner} won: {amount} : {loser_amount}")
                self.winner = winner

    def count_winner(self):
        white_chips = np.count_nonzero(self.board == 1)
        black_chips = np.count_nonzero(self.board == 2)
        if white_chips > black_chips:
            return 1, white_chips, black_chips
        elif white_chips < black_chips:
            return 2, black_chips, white_chips
        else:
            return 0, white_chips, black_chips  # draw

    def get_winner(self):
        return self.winner

    def __repr__(self):
        temp_board = np.copy(self.board)

        for move in self.valid_moves_to_reverse.keys():
            x, y = move
            temp_board[x, y] = 5
        return str(temp_board)


if __name__ == '__main__':
    pass
