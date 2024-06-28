import math
import os
import time

import numpy as np
from game_logic import Othello
from algorithms.minimax.minmax import (Minimax,
                                       fixed_depth_f,
                                       dynamic_depth_f)
from heuristics.heu1 import heuristic, heuristic2

from heuristics.ga.heu_func import (CountChips,
                                    CountDangerEarlyGame,
                                    CountCorners,
                                    MaximizeMyMoves,
                                    CountSaferEarlyGame)
from heuristics.ga.heu_func2 import (CountChips as CountChips2,
                                     CountDangerEarlyGame as CountDangerEarlyGame2,
                                     CountCorners as CountCorners2,
                                     MaximizeMyMoves as MaximizeMyMoves2)
from heuristics.ga.heu_ga import create_heuristic

from models.model_interface import ModelInterface


class MiniMaxAgent(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: Minimax = model

    def predict_best_move(self, game: Othello):
        action_probs = self.model.simulate(game)

        if self.deterministic or game.turn > 20:
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

human_heu = {#CountChips: CountChips(1.5),
             CountCorners: CountCorners(8, 2.5),
             CountDangerEarlyGame: CountDangerEarlyGame(5),
             MaximizeMyMoves: MaximizeMyMoves(100),
             }
ga_human = load_minimax_agent("human set",
                              fixed_depth_f(1),
                              # dynamic_depth_f,
                              create_heuristic(human_heu))

heu_params = {CountChips: CountChips(14.110659370384004),
              CountCorners: CountCorners(1.6725387766365891,
                                         2.7262102930984993),
              CountDangerEarlyGame: CountDangerEarlyGame(1.0988509935146311),
              MaximizeMyMoves: MaximizeMyMoves(946.9075924700552)}

ga_0 = load_minimax_agent("depth 1 GA",
                          fixed_depth_f(1),
                          create_heuristic(heu_params))

custom_heu = {CountCorners: CountCorners(1.141624324278253, 2.2005380364531657),
              CountDangerEarlyGame: CountDangerEarlyGame(1.5537486913611744),
              MaximizeMyMoves: MaximizeMyMoves(202.78118735939893)}
ga_1 = load_minimax_agent("depth 1 GA-custom heu",
                          fixed_depth_f(1),
                          create_heuristic(custom_heu))

custom_heu_2 = {CountCorners: CountCorners(1.5186058726512501, 2.5149087893017823),
                CountDangerEarlyGame: CountDangerEarlyGame(6.374267148895484),
                MaximizeMyMoves: MaximizeMyMoves(215.8041817188843)}
ga_2 = load_minimax_agent("depth 1 GA-custom-2 heu",
                          fixed_depth_f(1),
                          create_heuristic(custom_heu_2))

vpn_5 = {CountCorners: CountCorners(1.6619678910885987, 2.1102043167782876),
         CountDangerEarlyGame: CountDangerEarlyGame(5.025284240347834),
         MaximizeMyMoves: MaximizeMyMoves(120.00812831528076)}
ga_vpn_5 = load_minimax_agent("depth dyn GA-vpn-5",
                              fixed_depth_f(1),
                              # dynamic_depth_f,
                              create_heuristic(vpn_5))

new_heu1 = {CountCorners: CountCorners(1.894404927981066, 1.3218629263416726),
            CountDangerEarlyGame: CountDangerEarlyGame(1.0433194853622616),
            MaximizeMyMoves: MaximizeMyMoves(125.60263840811217),
            CountSaferEarlyGame: CountSaferEarlyGame(1.2743229470941557)}
ga_new = load_minimax_agent("depth dyn GA-new",
                            fixed_depth_f(1),
                            # dynamic_depth_f,
                            create_heuristic(new_heu1))

heu2 = {CountCorners2: CountCorners2(1.345726455834431, 2.1073849836637555),
        CountDangerEarlyGame2: CountDangerEarlyGame2(5.214213407375362),
        MaximizeMyMoves2: MaximizeMyMoves2(88.59176079991997, 16.676433765717864)
        }
ga2_best = load_minimax_agent("depth dyn GA2-best",
                              fixed_depth_f(1),
                              # dynamic_depth_f,
                              create_heuristic(heu2))
