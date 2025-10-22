from reversi_game.game_logic import Othello
from reversi_game.algorithms.minimax.minmax import (Minimax,
                                       fixed_depth_f,
                                       dynamic_depth_f)
from reversi_game.heuristics.heu1 import heuristic, heuristic2

# from reversi_game.heuristics.ga.heu_func import (CountChips,
#                                     CountDangerEarlyGame,
#                                     CountCorners,
#                                     MaximizeMyMoves,
#                                     CountSaferEarlyGame)
# from reversi_game.heuristics.ga.heu_func2 import (CountChips as CountChips2,
#                                      CountSaferEarlyGame as CountSaferEarlyGame2,
#                                      CountDangerEarlyGame as CountDangerEarlyGame2,
#                                      CountCorners as CountCorners2,
#                                      MaximizeMyMoves as MaximizeMyMoves2,
#                                      WeightedPieceCounter as WeightedPieceCounter2)
from reversi_game.heuristics.ga.heu_func import (CountChips,
                                    CountSaferEarlyGame,
                                    CountDangerEarlyGame,
                                    CountCorners,
                                    MaximizeMyMoves,
                                    WeightedPieceCounter)
from reversi_game.heuristics.ga.heu_ga import create_heuristic

from reversi_game.agents.agent_interface import AgentInterface


class MiniMaxAgent(AgentInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: Minimax = model

    def _predict_best_move(self, det, game: Othello):
        action_probs = self.model.simulate(game)

        if det:
            best_action = self.model.best_moves()  # list of best moves if multiple have same eval
            return best_action, None
        else:
            best_action = self.choose_stochastic(action_probs)  # inheritance
            return (best_action,), None


def load_minimax_agent(name, depth_f, heu_f):
    minimax_agent = Minimax(depth_f, heu_f)
    return MiniMaxAgent(f'MinMax {name}', minimax_agent)


human_heu_f = lambda: {CountChips: CountChips(start_turn=50),
                       CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
                                                  end_turn=50),
                       CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
                       CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
                       MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
                       WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
                       }
minmax_human_depth_1 = load_minimax_agent("minmax human - depth 1",
                                          fixed_depth_f(1),
                                          # dynamic_depth_f,
                                          create_heuristic(human_heu_f()))
minmax_human_depth_dyn = load_minimax_agent("minmax human depth dynamic",
                                            dynamic_depth_f,
                                            create_heuristic(human_heu_f()))

human_heu_f = lambda: {CountChips: CountChips(start_turn=50),
                       CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
                                                  end_turn=50),
                       CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
                       CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
                       # MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
                       WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
                       }
no_mmm = load_minimax_agent("no mmm",
                            fixed_depth_f(1),
                            # dynamic_depth_f,
                            create_heuristic(human_heu_f()))

human_heu_f = lambda: {CountChips: CountChips(start_turn=50),
                       CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
                                                  end_turn=50),
                       CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
                       CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
                       MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
                       # WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
                       }
no_wpc = load_minimax_agent("no wpc",
                            fixed_depth_f(1),
                            # dynamic_depth_f,
                            create_heuristic(human_heu_f()))

human_heu_f = lambda: {CountChips: CountChips(start_turn=50),
                       CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
                                                  end_turn=50),
                       CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
                       CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
                       # MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
                       # WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
                       }
no__mmm_wpc = load_minimax_agent("no wpc mmm",
                                 fixed_depth_f(1),
                                 # dynamic_depth_f,
                                 create_heuristic(human_heu_f()))

human_heu_f = lambda: {  # CountChips: CountChips(start_turn=50),
    # CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
    #                           end_turn=50),
    # CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
    # CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
    MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
    WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
}
mmm_wpc = load_minimax_agent("only wpc mmm",
                             fixed_depth_f(1),
                             # dynamic_depth_f,
                             create_heuristic(human_heu_f()))

human_heu_f = lambda: {  # CountChips: CountChips(start_turn=50),
    # CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
    #                           end_turn=50),
    CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
    CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
    MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
    WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
}
no__chips__corner = load_minimax_agent("no chips corner",
                                       fixed_depth_f(1),
                                       # dynamic_depth_f,
                                       create_heuristic(human_heu_f()))

human_heu_f = lambda: {CountChips: CountChips(start_turn=50),
                       CountCorners: CountCorners(corner_divisor=5, corner_exponent=2.5,
                                                  end_turn=50),
                       # CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5, end_turn=20),
                       # CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=5, end_turn=20),
                       MaximizeMyMoves: MaximizeMyMoves(max_score=100, ratio=10),
                       WeightedPieceCounter: WeightedPieceCounter(max_turn=50),
                       }
no__safe__danger = load_minimax_agent("no safe danger",
                                      fixed_depth_f(1),
                                      # dynamic_depth_f,
                                      create_heuristic(human_heu_f()))

# ---------------------------------- GA ----------------------------------------------

### nakon 1000 rundi tournament u geneskim algoritmima
# CountCorners: corner divisor: 1.1660590428529272, corner exponent: 2.38034446924993, end_turn: 56--,
# WeightedPieceCounter: max turn: 59--,
# CountChips: start_turn: 54--,
# MaximizeMyMoves: max my score: 44.27376301907929, max ratio div: 15.749362953723114--,
# CountSaferEarlyGame: safer divisor: 7.606340476247734, end_turn: 26--,
# id: 549863,
# score: 298/450

heu2_f = lambda: {
    CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=7.606, end_turn=26),
    WeightedPieceCounter: WeightedPieceCounter(max_turn=59),
    MaximizeMyMoves: MaximizeMyMoves(max_score=44.27, ratio=15.75),
    CountChips: CountChips(start_turn=54),
    CountCorners: CountCorners(corner_divisor=1.166, corner_exponent=2.38, end_turn=56)}

minmax_ga_best_depth_1 = load_minimax_agent("minmax GA depth 1",
                                            fixed_depth_f(1),
                                            create_heuristic(heu2_f()))

minmax_ga_depth_dyn = load_minimax_agent("minmax GA depth dyn",
                                         dynamic_depth_f,
                                         create_heuristic(heu2_f()))

heu2_f = lambda: {
    CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=8.15, end_turn=23),
    WeightedPieceCounter: WeightedPieceCounter(max_turn=58),
    MaximizeMyMoves: MaximizeMyMoves(max_score=37, ratio=17.24),
    CountChips: CountChips(start_turn=54),
    CountDangerEarlyGame: CountDangerEarlyGame(danger_mult=4.643, end_turn=10),
    CountCorners: CountCorners(corner_divisor=1.093, corner_exponent=2.528, end_turn=52)}

minmax_ga_best_depth_1_3 = load_minimax_agent("3nd ga",
                                              fixed_depth_f(1),
                                              create_heuristic(heu2_f()))

heu2_f = lambda: {
    CountSaferEarlyGame: CountSaferEarlyGame(safer_divisor=5.145, end_turn=23),
    WeightedPieceCounter: WeightedPieceCounter(max_turn=55),
    MaximizeMyMoves: MaximizeMyMoves(max_score=47.67, ratio=14.326),
    CountChips: CountChips(start_turn=49),
    CountCorners: CountCorners(corner_divisor=8.86, corner_exponent=2.55, end_turn=55)}

minmax_ga_best_depth_1_2 = load_minimax_agent("2nd ga",
                                              fixed_depth_f(1),
                                              create_heuristic(heu2_f()))
