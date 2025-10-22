import os
import random
import time
from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from gymnasium.spaces import Discrete, Box

from reversi_game.agents.agent_interface import AgentInterface
from reversi_game.game_logic import Othello


def action_masks(game):
    valid_moves = game.valid_moves()
    mask = np.zeros(game.board.shape, dtype=bool)
    # Set True for each index in the set
    for index in valid_moves:
        mask[index] = True
    return mask.flatten()


class MaskedPPOWrapper(AgentInterface):
    def __init__(self, name, model, use_cnn=False):
        super().__init__(name)
        self.use_cnn = use_cnn
        self.model: MaskablePPO = model
        if use_cnn:
            self.obs_space = Box(low=0, high=255, shape=(3, 8, 8), dtype=np.uint8)
        else:
            self.obs_space = Box(low=0, high=1, shape=(64 * 3,), dtype=np.float32)

    def _predict_best_move(self, det, game: Othello):
        if self.use_cnn:
            encoded_state = game.get_encoded_state_as_img()
        else:
            encoded_state = game.get_encoded_state().reshape(-1)  # for Mlp

        action, _ = self.model.predict(encoded_state,
                                       action_masks=action_masks(game),
                                       deterministic=det)

        (row, col) = Othello.get_decoded_field(action)  # from [0, 63] -> (0-7, 0-7)
        move = (int(row), int(col))  # some bug it return np.int64 idk who cares
        # print(f'action : {action}, - {action_game}')
        return (move,), None


def load_sb3_agent(name, model_location, cls=MaskablePPO, cnn=False, policy_cls=None):
    '''for ppo cnn it doesnt pickle policy class BUG, so you need to supply it'''
    custom_objects = {'lr_schedule': lambda _: 0.0005,  # only cuz of warnings...
                      'learning_rate': 0.0005,
                      'clip_range': 0.2,
                      'action_space': Discrete(64),
                      'seed': int(time.time())
                      }
    if policy_cls is not None:
        custom_objects['policy_class'] = policy_cls
    model = cls.load(model_location, custom_objects=custom_objects)
    return MaskedPPOWrapper(name, model, cnn)
