import numpy as np
import numba
from numba import types, njit

DIRECTIONS = np.array([(-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1)])


@njit
def create_empty_list_int64():
    alist = [(1, 2)]
    alist.clear()
    return alist


@njit
def njit_get_reversed_fields(board, player_turn, field):  # slowest part of the code
    res = []#create_empty_list_int64()

    row, col = field
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for x, y in directions:
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
                        res.append((row + x * i, col + y * i))
                        # res.append(row + x * i)
                        # res.append(col + y * i)
                break
    return res


@njit
def create_tuple_array(size):
    # flat_array = np.full(size * 2, -1, dtype=np.int64)  # Create an array of size * 2, filled with -1
    # tuple_array = flat_array.reshape((-1, 2))  # Reshape into a 2D array where each row represents a tuple
    # return tuple_array
    return np.array([(-1, -1)] * size)


@njit
def njit_get_all_reversed_fields(board, player_turn, fields):  # TODO: maybe make multiple smaller njit functions
    res = []  # create_empty_list_int64()
    sub_res = []  # create_empty_list_int64()

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
                            sub_res.append((row + x * i, col + y * i))
                            # res.append(row + x * i)
                            # res.append(col + y * i)
                    break
        if len(sub_res):
            res.append((-1, -1))
            res.append((row, col))
            res.extend(sub_res)
            sub_res.clear()
    return res


@njit
def slower_njit_get_all_reversed_fields(board, player_turn, fields):  # TODO: maybe make multiple smaller njit functions
    res = create_tuple_array(22 * len(fields))
    res_index = 0
    sub_res = create_tuple_array(20)
    sub_index = 0

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
                            sub_res[sub_index][0] = row + x * i  # (row + x * i, col + y * i)
                            sub_res[sub_index][1] = col + y * i
                            sub_index += 1
                            # res.append(row + x * i)
                            # res.append(col + y * i)
                    break
        if len(sub_res):
            # res.append((-1, -1))
            res_index += 1  # to let inbetween (-1, -1)
            res[res_index][0] = row
            res[res_index][1] = col
            res_index += 1
            # res.extend(sub_res)

            to_add = sub_res[:sub_index]
            res[res_index: res_index + len(to_add)] = to_add
            res_index += sub_index

            # sub_res.clear()
            sub_index = 0

    return res[:res_index]


class Othello:
    DIRECTIONS = {(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)}

    CORNERS = {(0, 0), (7, 7), (0, 7), (7, 0)}

    def __init__(self, players=("w", "b"), turn=1, board=None,
                 first_move=1, last_move=None, edge_fields=None, chips=(2, 2)):
        self.white, self.black = players
        self.player_turn = first_move  # possible turns {1, 2}, white is 1
        self.last_turn = last_move
        self.valid_moves_to_reverse = None
        self.winner = None
        self.turn = turn
        self.chips = chips
        if board is not None:
            self.board = np.array(board)
        else:
            self._set_default_board()

        if edge_fields is None:
            self.edge_fields = self._get_edge_fields()
        else:
            self.edge_fields = edge_fields
        self._calculate_next_valid_moves()
        self._check_correctness()

    def get_snapshot(self):
        return Othello(players=(self.white, self.black),
                       board=np.copy(self.board),
                       turn=self.turn,
                       first_move=self.player_turn,
                       last_move=self.last_turn,
                       edge_fields=self.edge_fields.copy(),
                       chips=self.chips)

    def _swap_player_turn(self):
        self.player_turn = 3 - self.player_turn

    def _set_default_board(self):
        self.board = np.full((8, 8), 0)
        self.board[3:5, 3:5] = [[1, 2],
                                [2, 1]]

    def valid_moves(self):
        return set(self.valid_moves_to_reverse.keys())

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

    def ALL_calculate_next_valid_moves(self):  # TODO make test to confirm old and new give same results
        moves_to_reverse = {}
        list_of_fields = []
        if len(self.edge_fields):
            edge_fields = np.array(list(self.edge_fields))
            list_of_fields = njit_get_all_reversed_fields(self.board,
                                                           self.player_turn,
                                                           edge_fields)

        idx = 1  # 0 is (-1, -1) if its not empty
        while idx < len(list_of_fields):
            field = list_of_fields[idx]#tuple(list_of_fields[idx])
            to_reverse = set()
            idx += 1

            while idx < len(list_of_fields) and list_of_fields[idx][0] != -1:
                to_reverse.add(list_of_fields[idx])
                idx += 1

            moves_to_reverse[field] = to_reverse
            idx += 1

        self.valid_moves_to_reverse = moves_to_reverse

    def _calculate_next_valid_moves(self):
        moves_to_reverse = {}
        for field in self.edge_fields:  # ... in self._get_edge_fields():
            if to_reverse := self._get_reversed_fields(field):
                moves_to_reverse[field] = to_reverse
        self.valid_moves_to_reverse = moves_to_reverse

    def _get_reversed_fields(self, field):  # slowest part of the code
        res = njit_get_reversed_fields(self.board, self.player_turn, field)
        res = set(res)

        # old = self.old_get_reversed_fields(field)
        # if res != old:
        #     print(f'res = {res}\nold = {old}\n')
        return res

    def old_get_reversed_fields(self, field):  # slowest part of the code
        s = set()
        maybe_s = set()
        row, col = field

        for x, y in Othello.DIRECTIONS:
            for times in range(1, 8):
                new_row = row + x * times
                new_col = col + y * times

                if not (0 <= new_row <= 7 and 0 <= new_col <= 7):
                    break

                match self.board[new_row, new_col]:
                    case 0:
                        break
                    case self.player_turn:
                        if times == 1:
                            break
                        else:
                            s = s.union(maybe_s)
                            break
                            # s.update(maybe_s)
                    case _:  # opponent case
                        maybe_s.add((new_row, new_col))
            # maybe_s.clear()
            maybe_s = set()

        return s

    def _get_edge_fields(self):
        player_positions = np.where(self.board != 0)#3 - self.player_turn)  # Get positions of opponents chips

        empty_adjacent = set()

        for i, j in zip(player_positions[0], player_positions[1]):
            neighbor_slice = self.board[max(0, i - 1):min(8, i + 2), max(0, j - 1):min(8, j + 2)]
            empty_spots = np.argwhere(neighbor_slice == 0) + np.array([max(0, i - 1), max(0, j - 1)])

            empty_adjacent.update({tuple(fields) for fields in empty_spots})
            # for spot in empty_spots:  # TODO  is this needed?
            #     x, y = spot
            #     if np.any(self.board[max(0, x - 1):min(8, x + 2), max(0, y - 1):min(8, y + 2)] != 0):
            #         empty_adjacent.add(tuple(spot))

        return empty_adjacent

    def play_move(self, field):
        if not self.winner:  # if the game haven't ended, you can play
            swap_value = self.player_turn

            if field in self.valid_moves():
                to_reverse = list(self.valid_moves_to_reverse[field])
                len_to_reverse = len(to_reverse)
                to_reverse.append(field)
                xs = np.array([x[0] for x in to_reverse], dtype=int)
                ys = np.array([x[1] for x in to_reverse], dtype=int)

                self.board[xs, ys] = np.array([swap_value] * len(to_reverse), dtype=int)
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
        i, j = field
        neighbor_slice = self.board[max(0, i - 1):min(8, i + 2), max(0, j - 1):min(8, j + 2)]
        empty_spots = np.argwhere(neighbor_slice == 0) + np.array([max(0, i - 1), max(0, j - 1)])

        self.edge_fields.update({tuple(fields) for fields in empty_spots})
        # for spot in empty_spots:  # TODO  is this needed?
        #     x, y = spot
        #     if np.any(self.board[max(0, x - 1):min(8, x + 2), max(0, y - 1):min(8, y + 2)] != 0):
        #         self.edge_fields.add((x, y))

    def _check_correctness(self):
        if not self.valid_moves_to_reverse:
            self._update_state()
            if not self.valid_moves_to_reverse:
                self.winner, _, _ = self.count_winner()

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

    def __repr__(self):
        temp_board = np.copy(self.board)

        for move in self.valid_moves_to_reverse.keys():
            x, y = move
            temp_board[x, y] = 5
        return str(temp_board)


if __name__ == '__main__':
    pass
