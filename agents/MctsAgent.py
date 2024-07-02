import math
import os
import time

import numpy as np
from game_logic import Othello
from algorithms.mcts.montecarlo import MCTS
from agents.agent_interface import AgentInterface


class MctsAgent(AgentInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: MCTS = model

    def predict_best_move(self, game: Othello):
        action_probs = self.model.simulate(game)

        if self.deterministic or game.turn > 10:  # No need to change to non deter, since its randomly simulating games, and wont play same moves if other player/agents plays same moves.
            best_action = self.model.best_moves()
            return best_action, None
        else:
            best_action = self.choose_stochastic(action_probs)
            return (best_action,), None

    # @staticmethod
    # def choose_stochastic(action_prob):
    #     encoded_action = np.random.choice(len(action_prob), p=action_prob)
    #     return Othello.get_decoded_field(encoded_action)


def load_mcts_model(params=None):

    if params is None:
        params = {}

    time_limit = params.get('max_time', math.inf)
    iter_limit = params.get('max_iter', 100)
    c = params.get('c', 1.41)
    verbose = params.get('verbose', 0)  # 0 means no logging

    mcts_model = MCTS(max_time=time_limit,
                      max_iter=iter_limit,
                      uct_exploration_const=c,
                      verbose=verbose)

    return MctsAgent(f'Mcts iter_limit {iter_limit}', mcts_model)

