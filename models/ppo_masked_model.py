from sb3_contrib.ppo_mask import MaskablePPO
from models.model_interface import ModelInterface
from game_logic import Othello
import numpy as np
import gymnasium.spaces as spaces
from gymnasium.spaces import Discrete, Box, Dict

from collections import OrderedDict


# class CustomPPOInterface(MaskablePPO, ModelInterface):
#
#     def predict(self, game):
#         pass

class MaskedPPOWrapper(ModelInterface):
    def __init__(self, model):
        self.model: MaskablePPO = model
        self.obs_space = Dict({
            'board': Box(0, 2, shape=(8, 8), dtype=int),
            'player': Discrete(2, start=1)
        })

    def action_masks(self, game):
        valid_moves = game.valid_moves()

        mask = np.zeros(game.board.shape, dtype=bool)

        # Set True for each index in the set
        for index in valid_moves:
            mask[index] = True
        mask.flatten()
        return mask

    def predict_best_move(self, game: Othello):
        obs = OrderedDict({
            'board': game.board,
            'player': game.player_turn
        })
        flattened_obs = spaces.flatten(self.obs_space, obs)

        action, _ = self.model.predict(flattened_obs,
                                       action_masks=self.action_masks(game),
                                       deterministic=False)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


def action_masks(game):
    valid_moves = game.valid_moves()
    mask = np.zeros(game.board.shape, dtype=bool)
    # Set True for each index in the set
    for index in valid_moves:
        mask[index] = True
    mask.flatten()
    return mask


class MaskedPPOWrapper2(ModelInterface):
    def __init__(self, model):
        self.model: MaskablePPO = model

    def predict_best_move(self, game: Othello):
        flattened_board = game.board.flatten()
        flattened_obs = np.append(flattened_board, game.player_turn)

        action, _ = self.model.predict(flattened_obs,
                                       action_masks=self.action_masks(game),
                                       deterministic=False)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


class MaskedPPOWrapper66(ModelInterface):
    def __init__(self, model):
        self.model: MaskablePPO = model
        self.obs_space = Dict({
            'board': Box(0, 2, shape=(8, 8), dtype=int),
            'num_chips': Discrete(129, start=-64),
            'player': Discrete(2, start=1)
        })

    def action_masks(self, game):
        valid_moves = game.valid_moves()

        mask = np.zeros(game.board.shape, dtype=bool)

        # Set True for each index in the set
        for index in valid_moves:
            mask[index] = True
        mask.flatten()
        return mask

    def get_chips_diff(self, game):
        ai = game.player_turn - 1
        return game.chips[ai] - game.chips[1 - ai]

    def predict_best_move(self, game: Othello):
        obs = OrderedDict({
            'board': game.board,
            'num_chips': self.get_chips_diff(game),
            'player': game.player_turn
        })
        flattened_obs = spaces.flatten(self.obs_space, obs)

        action, _ = self.model.predict(flattened_obs,
                                       action_masks=self.action_masks(game),
                                       deterministic=False)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


class MaskedPPOWrapper129(ModelInterface):
    def __init__(self, model):
        self.model: MaskablePPO = model
        self.obs_space = Dict({
            'board': Box(0, 2, shape=(8, 8), dtype=int),
            'mask': Box(0, 1, shape=(8, 8), dtype=int),
            'player': Discrete(2, start=1)
        })

    def action_masks(self, game):
        valid_moves = game.valid_moves()

        mask = np.zeros(game.board.shape, dtype=bool)

        # Set True for each index in the set
        for index in valid_moves:
            mask[index] = True
        mask.flatten()
        return mask

    def predict_best_move(self, game: Othello):
        obs = OrderedDict({
            'board': game.board,
            'mask': self.action_masks(game),
            'player': game.player_turn
        })
        flattened_obs = spaces.flatten(self.obs_space, obs)

        action, _ = self.model.predict(flattened_obs,
                                       action_masks=self.action_masks(game),
                                       deterministic=False)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


def load_model(file):
    model = MaskablePPO.load(file)
    return MaskedPPOWrapper(model)


def load_model_2(file):
    model = MaskablePPO.load(file)
    return MaskedPPOWrapper2(model)


def load_model_66(file):
    model = MaskablePPO.load(file)
    return MaskedPPOWrapper66(model)


def load_model_64_64_1(file):
    model = MaskablePPO.load(file)
    return MaskedPPOWrapper129(model)
