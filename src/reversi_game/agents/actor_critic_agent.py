import numpy as np
import torch

from reversi_game.game_logic import Othello
from reversi_game.algorithms.alphazero.AlphaZero import load_model
from reversi_game.agents.agent_interface import AgentInterface

class ACAgent(AgentInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model = model

    def _predict_best_move(self, det, game: Othello):
        # action_probs, value = self.model.forward(game.get_encoded_state())
        encoded_state = game.get_encoded_state()
        policy, _ = self.model(
            torch.tensor(encoded_state, device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).detach().numpy()

        valid_moves = game.valid_moves_encoded()
        policy *= valid_moves
        policy /= np.sum(policy)

        action_probs = policy

        if det:
            indices_of_max = np.where(action_probs == np.amax(action_probs))[0]

            # Convert indices to a list
            best_action = indices_of_max.tolist()
            best_action = [game.get_decoded_field(action) for action in best_action]
            return best_action, None
        else:
            best_action = self.choose_stochastic(action_probs)
            return (best_action,), None


def load_ac_agent(name, model_location=None, params=None):
    if model_location is None:
        raise Exception('ac model or file path needs to be supplied!!!')

    if params is None:
        params = {}

    hidden_layer = params.get('hidden_layer', 64)
    model = load_model(model_location, hidden_layer)
    model.eval()

    return ACAgent(f'actor-critic - {name}', model)