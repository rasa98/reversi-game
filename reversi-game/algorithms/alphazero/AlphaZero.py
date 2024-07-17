import numpy as np
import math
import random
import torch
import torch.nn.functional as F
from tqdm import trange

import sys
import os

if __name__ == '__main__' and os.environ['USER'] != 'student':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
    sys.path.append(source_dir)
# ---------------------
from game_logic import Othello, ALL_FIELDS_SIZE
from algorithms.alphazero.alpha_mcts import MCTS
from algorithms.alphazero.utils.neural_net import Net


class AlphaZero:
    def __init__(self, model, optimizer, params, mcts_params):
        self.model = model
        self.optimizer = optimizer
        self.params = params

        self.mcts = MCTS("alpha-mcts", model, **mcts_params)

    def self_play(self):
        data = []
        game = Othello()

        while True:
            player = game.player_turn  # !!! changed get_encoded_state
            encoded_perspective_state = game.get_encoded_state()
            action_probs, _ = self.mcts.simulate(game)

            data.append((encoded_perspective_state, action_probs, player))

            temp_action_probs = action_probs ** (1 / self.params['temp'])
            temp_action_probs /= np.sum(temp_action_probs)

            action = np.random.choice(ALL_FIELDS_SIZE, p=temp_action_probs)

            game.play_move(Othello.get_decoded_field(action))
            if game.is_game_over():
                data_to_return = []
                winner = game.get_winner()
                for state, action_probs, player in data:
                    value = 0
                    if winner == player:
                        value = 1
                    elif winner != 0:  # if its not a draw
                        value = -1

                    data_to_return.append((
                        state,
                        action_probs,
                        value
                    ))
                return data_to_return

    def train(self, data):
        random.shuffle(data)
        for batchIdx in range(0, len(data), self.params['batch_size']):
            sample = data[batchIdx:min(len(data) - 1, batchIdx + self.params[
                'batch_size'])]  # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        folder = 'alpha-zero'
        os.makedirs(folder, exist_ok=True)
        folder = f'{folder}/train-{len(os.listdir(folder)) + 1}'
        os.makedirs(folder, exist_ok=True)

        for iteration in range(self.params['num_iterations']):
            memory = []

            self.model.eval()
            for _ in trange(self.params['num_self_play_iterations']):
                memory += self.self_play()

            self.model.train()
            for _ in trange(self.params['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"{folder}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"{folder}/optimizer_{iteration}.pt")

            self.model.iterations_trained += 1


def load_model(model_path, hidden_layer_number, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net(hidden_layer_number, device)
    # model = ResNet(res_blocks, hidden_layer_number, device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


if __name__ == "__main__":
    import timeit

    game = Othello()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
    params = {
        'num_iterations': 200,
        'num_self_play_iterations': 20,
        'num_epochs': 10,
        'batch_size': 64,
        'temp': 1.1
    }
    mcts_params = {
        'uct_exploration_const': 1.3,
        'max_iter': 30,
        # these are flexible dirichlet epsilon for noise
        # favor exploration more in the beginning
        'dirichlet_epsilon': 0.40,
        'initial_alpha': 0.6,
        'final_alpha': 0.25,
        'decay_steps': 200
    }
    azero = AlphaZero(model, optimizer, params, mcts_params)

    execution_time = timeit.timeit(azero.learn, number=1)
    print(f"Execution time: {execution_time} seconds")

    # move_to_testing()
