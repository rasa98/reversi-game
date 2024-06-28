import os
import time

from sb3_contrib.ppo_mask import MaskablePPO
from models.model_interface import ModelInterface
from game_logic import Othello
import numpy as np
import gymnasium.spaces as spaces
from gymnasium.spaces import Discrete, Box, Dict, MultiDiscrete

from collections import OrderedDict


# class CustomPPOInterface(MaskablePPO, ModelInterface):
#
#     def predict(self, game):
#         pass

class MaskedPPOWrapper(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
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
                                       deterministic=self.deterministic)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


def action_masks(game):
    valid_moves = game.valid_moves()
    mask = np.zeros(game.board.shape, dtype=bool)
    # Set True for each index in the set
    for index in valid_moves:
        mask[index] = True
    return mask.flatten()


class MaskedPPOWrapper2(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: MaskablePPO = model

    def predict_best_move(self, game: Othello):
        flattened_board = game.board.flatten()
        flattened_obs = np.append(flattened_board, game.player_turn)

        action, _ = self.model.predict(flattened_obs,
                                       action_masks=action_masks(game),
                                       deterministic=self.deterministic)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


class MaskedPPOWrapper66(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
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
                                       deterministic=self.deterministic)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


class MaskedPPOWrapper129(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
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
                                       deterministic=self.deterministic)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


class MaskedPPOWrapper64_2_1(ModelInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: MaskablePPO = model
        self.obs_space = Dict({
            'board': Box(0, 2, shape=(8, 8), dtype=int),
            'chips': MultiDiscrete([65, 65]),
            'player': Discrete(2, start=1)
        })

    def predict_best_move(self, game: Othello):
        obs = OrderedDict({
            'board': game.board,
            'chips': np.array(game.chips),
            'player': game.player_turn
        })
        flattened_obs = spaces.flatten(self.obs_space, obs)

        action, _ = self.model.predict(flattened_obs,
                                       action_masks=action_masks(game),
                                       deterministic=self.deterministic)

        action_game = (action // 8, action % 8)
        # print(f'action : {action}, - {action_game}')
        return (action_game,), None


class MaskedPPOWrapperNew(ModelInterface):
    def __init__(self, name, model, use_cnn=False):
        super().__init__(name)
        self.use_cnn = use_cnn
        self.model: MaskablePPO = model
        if use_cnn:
            self.obs_space = Box(low=0, high=255, shape=(3, 8, 8), dtype=np.uint8)
        else:
            self.obs_space = Box(low=0, high=1, shape=(64 * 3,), dtype=np.float32)

    def predict_best_move(self, game: Othello):
        if self.use_cnn:
            encoded_state = game.get_encoded_state_as_img()
        else:
            encoded_state = game.get_encoded_state().reshape(-1)  #  for Mlp
        det = self.deterministic
        if game.turn > 20:
            det = True
        action, _ = self.model.predict(encoded_state,
                                       action_masks=action_masks(game),
                                       deterministic=det)

        move = Othello.get_decoded_field(action)  # from [0, 63] -> (0-7, 0-7)
        # print(f'action : {action}, - {action_game}')
        return (move,), None


def load_sb3_model(name, file, cls=MaskablePPO, cnn=False, policy_cls=None):  # TODO generalize this module
    '''for ppo cnn it doesnt pickle policy class BUG, so you need to supply it'''
    custom_objects = {'lr_schedule': lambda _: 0.0005,  # only cuz of warnings...
                      'learning_rate': 0.0005,
                      'clip_range': 0.2,
                      'action_space': Discrete(64),
                      'seed': int(time.time())
                      }
    if policy_cls is not None:
        custom_objects['policy_class'] = policy_cls
    model = cls.load(file, custom_objects=custom_objects)
    return MaskedPPOWrapperNew(name, model, cnn)


def load_model(name, file):
    model = MaskablePPO.load(file, custom_objects={'seed': int(time.time())})
    return MaskedPPOWrapper(name, model)


def load_model_2(name, file):
    model = MaskablePPO.load(file, custom_objects={'seed': int(time.time())})
    return MaskedPPOWrapper2(name, model)


def load_model_66(name, file):
    model = MaskablePPO.load(file, custom_objects={'seed': int(time.time())})
    return MaskedPPOWrapper66(name, model)


def load_model_64_64_1(name, file):
    model = MaskablePPO.load(file, custom_objects={'seed': int(time.time())})
    return MaskedPPOWrapper129(name, model)


def load_model_64_2_1(name, filepath):
    model = MaskablePPO.load(filepath, custom_objects={'seed': int(time.time())})
    return MaskedPPOWrapper64_2_1(name, model)

# file = 'training/rl/Dict_obs_space/history_00000385'
# ai385 = load_model('ppo_masked_385', file)
#
# file = 'training/rl/Dict_obs_space/mppo_num_chips/models/history_00000330'
# fixed_330 = load_model_64_2_1('fixed_ppo_masked_330', file)
