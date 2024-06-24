import math
import os
import time

import numpy as np
from game_logic import Othello
from algorithms.minimax.minmax import (Minimax,
                                       fixed_depth_f,
                                       dynamic_depth_f)
from heuristics.heu1 import heuristic, heuristic2
from heuristics.heu2 import create_heuristic
from heuristics.ga.heu_func import (CountChips,
                                    CountDangerEarlyGame,
                                    CountCorners,
                                    MaximizeMyMoves)
from heuristics.ga.heu_ga import create_heuristic as create_heuristic2

from models.model_interface import ModelInterface


class MiniMaxAgent(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: Minimax = model

    def predict_best_move(self, game: Othello):
        action_probs = self.model.simulate(game)

        if self.deterministic:  # No need to change to non deter, since its randomly simulating games, and wont play same moves if other player/agents plays same moves.
            best_action = self.model.best_moves()  # list of best moves if multiple have same eval
            return best_action, None
        else:
            best_action = self.choose_stochastic(action_probs)  # inheritance
            return (best_action,), None


def load_minimax_agent(name, depth_f, heu_f):
    minimax_agent = Minimax(depth_f, heu_f)
    return MiniMaxAgent(f'MinMax {name}', minimax_agent)


mm_static = load_minimax_agent('static depth',
                               lambda _: 3,
                               heuristic)
mm2_dynamic = load_minimax_agent('dyn depth',
                                 dynamic_depth_f,
                                 heuristic2)

heu_params = (14.110659370384004,
              1.6725387766365891,
              2.7262102930984993,
              1.0988509935146311,
              946.9075924700552)
ga_0 = load_minimax_agent("depth 1 GA",
                          fixed_depth_f(1),
                          create_heuristic(*heu_params))

custom_heu = {CountCorners: CountCorners(1.141624324278253, 2.2005380364531657),
              CountDangerEarlyGame: CountDangerEarlyGame(1.5537486913611744),
              MaximizeMyMoves: MaximizeMyMoves(202.78118735939893)}
ga_1 = load_minimax_agent("depth 1 GA-custom heu",
                          fixed_depth_f(1),
                          create_heuristic2(custom_heu))

custom_heu_2 = {CountCorners: CountCorners(1.5186058726512501, 2.5149087893017823),
                CountDangerEarlyGame: CountDangerEarlyGame(6.374267148895484),
                MaximizeMyMoves: MaximizeMyMoves(215.8041817188843)}
ga_2 = load_minimax_agent("depth 1 GA-custom-2 heu",
                          fixed_depth_f(1),
                          create_heuristic2(custom_heu_2))

vpn_5 = {CountCorners: CountCorners(1.6619678910885987, 2.1102043167782876),
         CountDangerEarlyGame: CountDangerEarlyGame(5.025284240347834),
         MaximizeMyMoves: MaximizeMyMoves(120.00812831528076)}
ga_vpn_5 = load_minimax_agent("depth dyn GA-vpn-5",
                              # fixed_depth_f(1),
                              dynamic_depth_f,
                              create_heuristic2(vpn_5))

# mm_static = Minimax('MinMax static', lambda _: 3, heuristic)
# mm2_dynamic = Minimax('MinMax dyn', dynamic_depth_f, heuristic2)
#
# heu_params = (14.110659370384004,
#               1.6725387766365891,
#               2.7262102930984993,
#               1.0988509935146311,
#               946.9075924700552)
# ga_0 = Minimax("depth 1 GA", fixed_depth_f(1), create_heuristic(*heu_params))
#
# custom_heu = {CountCorners: CountCorners(1.141624324278253, 2.2005380364531657),
#               CountDangerEarlyGame: CountDangerEarlyGame(1.5537486913611744),
#               MaximizeMyMoves: MaximizeMyMoves(202.78118735939893)}
# ga_1 = Minimax("depth 1 GA-custom heu", fixed_depth_f(1), create_heuristic2(custom_heu))
#
# custom_heu_2 = {CountCorners: CountCorners(1.5186058726512501, 2.5149087893017823),
#                 CountDangerEarlyGame: CountDangerEarlyGame(6.374267148895484),
#                 MaximizeMyMoves: MaximizeMyMoves(215.8041817188843)}
# ga_2 = Minimax("depth 1 GA-custom-2 heu", fixed_depth_f(1), create_heuristic2(custom_heu_2))
#
# # vpn cluster - folder 5 -
# # min_opp_score: 120.00812831528076,
# # corner divisor: 1.6619678910885987,
# # corner exponent: 2.1102043167782876,
# # danger divisor: 5.025284240347834
# vpn_5 = {CountCorners: CountCorners(1.6619678910885987, 2.1102043167782876),
#          CountDangerEarlyGame: CountDangerEarlyGame(5.025284240347834),
#          MaximizeMyMoves: MaximizeMyMoves(120.00812831528076)}
# ga_vpn_5 = Minimax("depth dyn GA-vpn-5",
#                    # fixed_depth_f(1),
#                    dynamic_depth_f,
#                    create_heuristic2(vpn_5))
