from abc import ABC, abstractmethod
from game_logic import Othello


class ModelInterface(ABC):
    def __init__(self, name):
        self.name = name
        self.deterministic = True

    def set_deterministic(self, det):
        self.deterministic = det
        return self

    @abstractmethod
    def predict_best_move(self, game: Othello):
        """Abstract method to make predictions."""
        pass

    def __str__(self):
        return self.name


class RandomModel(ModelInterface):
    def __init__(self):
        super().__init__('Random model')

    def predict_best_move(self, game: Othello):
        return list(game.valid_moves()), None


ai_random = RandomModel()
