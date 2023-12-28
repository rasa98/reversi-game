import unittest
import numpy as np
from some_boards import board1, next_white_moves_board1, next_black_moves_board1, edge1_white, edge1_black
from src.game_logic import Othello


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.b1_white = Othello(board=board1, first_move=1)
        self.b1_black = Othello(board=board1, first_move=2)

    def test_default_board(self):
        b = Othello()
        self.assertEqual(b.player_turn, 1)
        board_np = np.array(board1)
        self.assertEqual(b.board.all(), board_np.all())

    def test_empty_edges(self):
        self.assertEqual(self.b1_white._get_edge_fields(), edge1_white)
        self.assertEqual(self.b1_black._get_edge_fields(), edge1_black)

        self.assertNotEqual(self.b1_white._get_edge_fields(), edge1_black)
        self.assertNotEqual(self.b1_black._get_edge_fields(), edge1_white)

    def test_injected_board(self):
        self.assertEqual(self.b1_white.valid_moves(), next_white_moves_board1)
        self.assertEqual(self.b1_black.valid_moves(), next_black_moves_board1)


if __name__ == '__main__':
    unittest.main()
