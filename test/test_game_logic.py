import sys
sys.path.append('/home/rasa/PycharmProjects/reversi/')  # TODO fix this hack

import unittest
import numpy as np
from test.some_boards import board1, next_white_moves_board1, next_black_moves_board1, edge1_white, edge1_black
from game_logic import Othello


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.b1_white = Othello(first_move=1)
        self.b1_black = Othello(first_move=2)

    def test_default_board(self):
        b = Othello()
        self.assertEqual(b.player_turn, 1)
        board_np = np.array(board1)
        self.assertEqual(b.board.all(), board_np.all())

    # def test_empty_edges(self):   TODO: fix this
    #     self.assertEqual(self.b1_white._get_edge_fields(), edge1_white)
    #     self.assertEqual(self.b1_black._get_edge_fields(), edge1_black)
    #
    #     self.assertNotEqual(self.b1_white._get_edge_fields(), edge1_black)
    #     self.assertNotEqual(self.b1_black._get_edge_fields(), edge1_white)

    def test_injected_board(self):
        self.assertEqual(self.b1_white.valid_moves(), next_white_moves_board1)
        self.assertEqual(self.b1_black.valid_moves(), next_black_moves_board1)

    # TODO: test this
    # def _get_reversed_fields(self, field):  # slowest part of the code
    #     res = njit_get_reversed_fields(self.board, self.player_turn, field)
    #     # print(res)
    #     res = set((res[idx], res[idx + 1]) for idx in range(0, len(res), 2))
    #     old = self.old_get_reversed_fields(field)
    #     if res != old:
    #         print(f'res = {res}\nold = {old}\n')
    #     return res


if __name__ == '__main__':
    unittest.main()
