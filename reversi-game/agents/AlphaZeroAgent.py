import math
import os
import time

import numpy as np
from game_logic import Othello
from algorithms.alphazero.AlphaZero import load_model
from algorithms.alphazero.alpha_mcts import MCTS
from agents.agent_interface import AgentInterface
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


class AlphaZeroAgent(AgentInterface):
    def __init__(self, name, model):
        super().__init__(name)
        self.model: MCTS = model

    def predict_best_move(self, game: Othello):
        action_probs = self.model.simulate(game)
        if self.deterministic or game.turn > 10:
            best_action = self.model.best_moves()
            return best_action, None
        else:
            best_action = self.choose_stochastic(action_probs)
            return (best_action,), None


def load_azero_agent(name, file=None, model=None, params=None):
    if file is None and model is None:
        raise Exception('azero model or file path needs to be supplied!!!')

    if params is None:
        params = {}

    # res_blocks = params.get('res_blocks', 4)
    hidden_layer = params.get('hidden_layer', 128)
    time_limit = params.get('max_time', math.inf)
    iter_limit = params.get('max_iter', 100)
    c = params.get('uct_exploration_const', 1.41)
    dirichlet_epsilon = params.get('dirichlet_epsilon', 0)
    # initial_alpha = params.get('initial_alpha', 0.4)
    # final_alpha = params.get('final_alpha', 0.1)
    verbose = params.get('verbose', 0)  # 0 means no logging

    if model is None:
        model = load_model(file, hidden_layer)
        # model = load_model(file, res_blocks, hidden_layer)
    elif isinstance(model, MaskableActorCriticPolicy):
        model = WrapPPOPolicy(model)

    model.eval()

    mcts = MCTS(model,
                max_time=time_limit,
                max_iter=iter_limit,
                uct_exploration_const=c,
                dirichlet_epsilon=dirichlet_epsilon,
                verbose=verbose)

    return AlphaZeroAgent(f'alpha-mcts - {name}', mcts)


def model_generator(file_location, model_idxs, params):
    for i in model_idxs:
        model_location = f'{file_location}/model_{i}.pt'
        try:
            model = load_azero_agent(f'{i}',
                                     file=model_location,
                                     params=params)
        except FileNotFoundError as e:
            print(e)
            break
        yield model


def model_generator_all(file_location, params):
    cwd = os.getcwd()
    d = os.path.join(cwd, file_location)
    file_names = [f for f in os.listdir(d)
                  if os.path.isfile(os.path.join(d, f)) and f.startswith('model')]

    # In the meantime if new agents were created, also detect them
    previous_contents = set()

    while True:
        current_contents = file_names
        new_files = current_contents - previous_contents
        if new_files:
            for i, f in enumerate(sorted(list(new_files))):
                model_location = f'{file_location}/{f}'
                model = load_azero_agent(f'{f[:-3]}',
                                         file=model_location,
                                         params=params)
                yield model
        else:
            break

        previous_contents = current_contents


def multi_folder_load_models(folder_params):
    for folder, params in folder_params:
        print(f'\n++++++++++++++++ TESTED FOLDER - {folder}++++++++++++++++\n')
        yield from model_generator_all(folder, params)
        print()


def multi_folder_load_some_models(folder_idxs_params):
    for folder, model_idxs, params in folder_idxs_params:
        print(f'\n++++++++++++++++ TESTED FOLDER - {folder}++++++++++++++++\n')
        yield from model_generator(folder, model_idxs, params)
        print()


class WrapPPOPolicy:
    def __init__(self, policy):
        self.iterations_trained = 0
        self.policy = policy
        self.device = policy.device

    def forward(self, obs):
        features = self.policy.extract_features(obs)
        latent_pi, latent_vf = self.policy.mlp_extractor(features)
        values = self.policy.value_net(latent_vf)
        action_logits = self.policy.action_net(latent_pi)
        return action_logits, values

    def eval(self):
        self.policy.eval()

    def __call__(self, obs):
        return self.forward(obs)