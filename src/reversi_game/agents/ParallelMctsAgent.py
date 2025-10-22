import math

from reversi_game.game_logic import Othello
from reversi_game.algorithms.mcts.ParallelMCTS import PMCTS
from reversi_game.agents.agent_interface import AgentInterface


class ParallelMctsAgent(AgentInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: PMCTS = model

    def open_pool(self, num_process=-1):
        if num_process <= -1:
            self.model.open_pool()
        else:
            self.model.open_pool(num_process)

    def clean_pool(self):
        self.model.clean()

    def _predict_best_move(self, det, game: Othello):
        action_probs = self.model.simulate(game)

        if det:
            best_action = self.model.best_moves()
            return best_action, None
        else:
            best_action = self.choose_stochastic(action_probs)
            return (best_action,), None


def load_parallel_mcts_model(params=None):
    if params is None:
        params = {}

    time_limit = params.get('max_time', math.inf)
    iter_limit = params.get('max_iter', 100)
    c = params.get('c', 1.41)
    verbose = params.get('verbose', 0)  # 0 means no logging

    mcts_model = PMCTS(time_limit=time_limit,
                       iter_limit=iter_limit,
                       c=c,
                       verbose=verbose)

    return ParallelMctsAgent(f'Pmcts {iter_limit}', mcts_model)
