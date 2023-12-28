from abc import ABC, abstractmethod
from game_logic import Othello


class ModelInterface(ABC):

    @abstractmethod
    def predict_best_move(self, game: Othello):
        """Abstract method to make predictions."""
        pass