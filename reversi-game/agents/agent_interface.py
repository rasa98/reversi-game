from abc import ABC, abstractmethod
from game_logic import Othello
import numpy as np


class AgentInterface(ABC):
    def __init__(self, name):
        self.name = name
        self.deterministic = True
        self.action_probs = None

    def set_deterministic(self, det):
        self.deterministic = det
        return self

    def predict_best_move(self, game: Othello):
        """Abstract method to make predictions."""
        if game.turn == 1:
            return self.make_random_move(game)  # first move is symmetrical so its fine to random it
        return self._predict_best_move(game)

    @abstractmethod
    def _predict_best_move(self, game: Othello):
        """Abstract method to make predictions."""
        pass

    @staticmethod
    def make_random_move(game: Othello):
        return list(game.valid_moves()), None

    @staticmethod
    def choose_stochastic(action_prob):
        encoded_action = np.random.choice(len(action_prob), p=action_prob)
        return Othello.get_decoded_field(encoded_action)

    def __str__(self):
        return self.name


class RandomAgent(AgentInterface):
    def __init__(self):
        super().__init__('Random model')

    def _predict_best_move(self, game: Othello):
        return self.make_random_move(game)


ai_random = RandomAgent()
