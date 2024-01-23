import unittest
import sys
import os

source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(source_dir)

from models.montecarlo import MCTS


class MyTestCase(unittest.TestCase):
    # def setUp(self):
    #     self.b1_white = Othello(board=board1, first_move=1)
    #     self.b1_black = Othello(board=board1, first_move=2)

    # def test_meth(self):
    #     self.assertEqual(len(ss), len(res + swap_pairs))
    pass

    # def select_highest_ucb_child(self):
    #     max_ucb = -math.inf
    #     max_child = None
    #     log_visited = math.log(self.visited)
    #     for child in self.children:
    #         if max_ucb < child.get_uct(log_visited):
    #             max_child = child
    #
    #     max_child2 = max(self.children, key=lambda c: c.get_uct(log_visited))
    #     assert max_child == max_child2  #  TODO: check if its correct way to find max



if __name__ == '__main__':
    unittest.main()
