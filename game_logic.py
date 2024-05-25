import numpy as np
import numba
import copy
from numba import types, njit
from itertools import count

DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@njit(cache=True)
def njit_check_empty_edge_fields(board, field):
    target_row, target_col = field
    result = []

    for offset_row, offset_col in DIRECTIONS:
        neighbor_row, neighbor_col = target_row + offset_row, target_col + offset_col

        if 0 <= neighbor_row < board.shape[0] and 0 <= neighbor_col < board.shape[1]:
            if board[neighbor_row, neighbor_col] == 0:
                result.append((neighbor_row, neighbor_col))

    return result


@njit(cache=True)
def njit_get_all_reversed_fields(board, player_turn, fields):  # TODO: maybe make multiple smaller njit functions
    res = []
    to_reverse = []

    for field in fields:
        row, col = field
        for x, y in DIRECTIONS:
            for times in range(1, 8):
                new_row = row + x * times
                new_col = col + y * times

                if not (0 <= new_row <= 7 and 0 <= new_col <= 7):
                    break

                value = board[new_row, new_col]
                if value == 0:
                    break
                elif value == player_turn:
                    if times != 1:
                        for i in range(1, times):
                            to_reverse.append((row + x * i, col + y * i))
                    break
        if len(to_reverse):
            res.append((-1, -1))
            res.append((row, col))
            res.extend(to_reverse)

            to_reverse.clear()
    return res


class Othello:
    DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1))

    CORNERS = {(0, 0), (7, 7), (0, 7), (7, 0)}

    def __init__(self, players=("w", "b"), turn=1, board=None,
                 first_move=1, last_move=None, edge_fields=None,
                 valid_moves_to_reverse=None, valid_moves_size=None, chips=(2, 2),
                 winner=None):
        self.white, self.black = players
        self.player_turn = first_move  # possible turns {1, 2}, white is 1
        self.last_turn = last_move
        self.valid_moves_to_reverse = valid_moves_to_reverse
        self.valid_moves_size = valid_moves_size
        self.winner = winner
        self.turn = turn
        self.chips = chips
        # self.played_moves = played_moves  # TODO: remove played_moves, used for mcts reuse tree

        if board is not None:
            self.board = np.array(board)
        else:
            self._set_default_board()

        if edge_fields is None:
            self.edge_fields = {(2, 2), (2, 3), (2, 4), (2, 5),
                                (3, 2), (3, 5), (4, 2), (4, 5),
                                (5, 2), (5, 3), (5, 4), (5, 5)}

        else:
            self.edge_fields = edge_fields

        if valid_moves_to_reverse is None or valid_moves_size is None:
            self._calculate_next_valid_moves()
            self._check_correctness()

    def get_snapshot(self):
        return Othello(players=(self.white, self.black),
                       board=np.copy(self.board),
                       turn=self.turn,
                       first_move=self.player_turn,
                       last_move=self.last_turn,
                       edge_fields=self.edge_fields.copy(),
                       valid_moves_to_reverse=copy.deepcopy(self.valid_moves_to_reverse),
                       valid_moves_size=self.valid_moves_size,
                       chips=self.chips,
                       winner=self.winner)

    def _swap_player_turn(self):
        self.player_turn = 3 - self.player_turn

    def _set_default_board(self):
        self.board = np.full((8, 8), 0)
        self.board[3:5, 3:5] = [[1, 2],
                                [2, 1]]

    def valid_moves(self):
        return set(self.valid_moves_to_reverse.keys())

    def valid_moves_encoded(self):
        valid_moves_layer = np.zeros_like(self.board, dtype=np.uint8)
        for move in self.valid_moves():
            valid_moves_layer[move] = 1
        return valid_moves_layer.reshape(-1)

    @staticmethod
    def get_encoded_field(field):
        """input (0-7, 0-7)
           output 0-63"""
        row, col = field[0], field[1]
        return row * 8 + col

    @staticmethod
    def get_decoded_field(action):
        """input 0-63
           output (0-7, 0-7)"""
        row = action // 8
        col = action % 8
        return row, col

    def valid_moves_sorted(self):
        """
        valid moves/fields gets sorted by moves that turns the most chips
        and if it is a corner.
        Useful for a b pruning in minmax.
        """

        def sort_f(key_field):
            res = 0
            if key_field in Othello.CORNERS:
                res += 4
            return res + len(self.valid_moves_to_reverse[key_field])

        return sorted(self.valid_moves_to_reverse,
                      key=sort_f,
                      reverse=True)

    def update_chips(self, turned: int):
        #  every turn: 1 new, turned >= 1
        to_add = 1 + turned
        x = self.chips[self.player_turn - 1] + to_add
        y = self.chips[2 - self.player_turn] - turned
        if self.player_turn == 1:
            self.chips = (x, y)
        else:
            self.chips = (y, x)

    def _calculate_next_valid_moves(self):  # TODO make test to confirm old and new give same results
        moves_to_reverse = {}
        list_of_fields = []
        if len(self.edge_fields):
            edge_fields = np.array(list(self.edge_fields))
            list_of_fields = njit_get_all_reversed_fields(self.board,
                                                          self.player_turn,
                                                          edge_fields)

        idx = 1  # 0 is (-1, -1) if its not empty
        while idx < len(list_of_fields):
            field = list_of_fields[idx]
            to_reverse = set()
            idx += 1

            while idx < len(list_of_fields) and list_of_fields[idx][0] != -1:
                to_reverse.add(list_of_fields[idx])
                idx += 1

            moves_to_reverse[field] = to_reverse
            idx += 1

        self.valid_moves_to_reverse = moves_to_reverse
        self.valid_moves_size = len(self.valid_moves_to_reverse)

    def play_move(self, field):
        if self.winner is None:  # if the game haven't ended, you can play
            swap_value = self.player_turn

            if field in self.valid_moves():
                to_reverse = self.valid_moves_to_reverse[field]
                len_to_reverse = len(to_reverse)

                self.board[field] = swap_value
                for f in to_reverse:
                    self.board[f] = swap_value

                # self.played_moves.append(field)
                self.update_edge_fields(field)
                self.last_turn = self.player_turn
                self.turn += 1
                self.update_chips(len_to_reverse)
                self._update_state()
                self._check_correctness()
                return self.chips

    def _update_state(self):
        self._swap_player_turn()
        self._calculate_next_valid_moves()

    def update_edge_fields(self, field):
        self.edge_fields.remove(field)
        list_of_empty_fields = njit_check_empty_edge_fields(self.board, field)
        self.edge_fields.update(list_of_empty_fields)

    def _check_correctness(self):
        if self.valid_moves_size == 0:  # not self.valid_moves_to_reverse:
            # if self.valid_moves_size == 0:
            self._update_state()
            if self.valid_moves_size == 0:  # not self.valid_moves_to_reverse:
                self.winner, _, _ = self.count_winner()

    def count_winner(self):
        white_chips = self.chips[0]
        black_chips = self.chips[1]
        if white_chips > black_chips:
            return 1, white_chips, black_chips
        elif white_chips < black_chips:
            return 2, black_chips, white_chips
        else:
            return 0, white_chips, black_chips  # draw

    def get_winner(self):
        return self.winner

    def is_game_over(self):
        return self.get_winner() is not None

    def get_winner_and_print(self):
        winner, amount, loser_amount = self.count_winner()
        match winner:
            case 0:
                print(f"Its a draw - {amount} : {loser_amount}")
                return "draw"
            case 1:
                print(f"{self.white} won: {amount} : {loser_amount}")
                return self.white
            case 2:
                print(f"{self.black} won: {amount} : {loser_amount}")
                return self.black
            case _:
                print("Game is still playing!")

    # def get_encoded_state_valid(self):
    #     valid_moves_layer = np.zeros_like(self.board, dtype=np.float32)
    #     for move in self.valid_moves():
    #         valid_moves_layer[move] = 1.0
    #
    #     encoded_state = np.stack(
    #         (valid_moves_layer, self.board == 1, self.board == 2)
    #     ).astype(np.float32)
    #
    #     return encoded_state

    def get_encoded_state(self):
        """ from_perspective can be: 1 or 2 """
        state = self.board
        from_perspective = self.player_turn
        encoded_state = np.stack(
            (state == 0, state == from_perspective, state == 3 - from_perspective)
        ).astype(np.float32)

        return encoded_state

    def __repr__(self):
        temp_board = np.copy(self.board)

        for move in self.valid_moves_to_reverse.keys():
            x, y = move
            temp_board[x, y] = 5
        return str(temp_board)


if __name__ == '__main__':
    pass
