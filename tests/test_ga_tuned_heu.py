import random
import unittest
import numpy as np
import sys
sys.path.append('/home/rasa/PycharmProjects/reversi/')  # TODO fix this hack
from reversi_game.scripts.ga.cluster.ga_tuned_heu import generate_all_pairs


class MyTestCase(unittest.TestCase):
    # def setUp(self):
    #     self.b1_white = Othello(board=board1, first_move=1)
    #     self.b1_black = Othello(board=board1, first_move=2)

    def test_robin_round_pair_generating(self):
        """
        test if func returns unique pairings for every el of
        population, for both starting positions
        """
        for _ in range(100):
            population_size = random.randint(1, 100) * 2
            rounds = random.randint(1, population_size-1)

            players = list(range(population_size))
            random.shuffle(players)

            res = generate_all_pairs(players, rounds)
            swap_pairs = [(y, x) for (x, y) in res]
            ss = set(res + swap_pairs)

            self.assertEqual(len(ss), len(res + swap_pairs))


if __name__ == '__main__':
    unittest.main()
