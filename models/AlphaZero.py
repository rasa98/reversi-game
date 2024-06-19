import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# for teseting purposes
import sys
import os

source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(source_dir)
# ---------------------
from game_logic import Othello
from models.montecarlo_alphazero_version import MCTS

GAME_ROW_COUNT = 8
GAME_COLUMN_COUNT = 8
ALL_FIELDS_SIZE = GAME_ROW_COUNT * GAME_COLUMN_COUNT


class ResNet(nn.Module):
    def __init__(self, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.iterations_trained = 0

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, ALL_FIELDS_SIZE)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


# class ResNet(nn.Module):
#     def __init__(self, num_hidden, device):
#         super().__init__()
#         self.device = device
#         self.iterations_trained = 0
#
#         self.device = device
#
#         self.startBlock = nn.Sequential(
#             nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Dropout(p=0.3)  # Dropout layer
#         )
#
#         self.sharedConv = nn.Sequential(
#             nn.Conv2d(num_hidden, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Dropout(p=0.3),  # Dropout layer
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Dropout(p=0.3)  # Dropout layer
#         )
#
#         self.policyHead = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(512 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, ALL_FIELDS_SIZE)
#         )
#
#         self.valueHead = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(128 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, 1),
#             nn.Tanh()
#         )
#
#         self.to(device)
#
#     def forward(self, x):
#         x = self.startBlock(x)
#         shared_out = self.sharedConv(x)
#         policy = self.policyHead(shared_out)
#         value = self.valueHead(shared_out)
#         return policy, value


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
            action_probs = self.mcts.simulate(game)

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


def load_model(model_path, res_blocks, hidden_layer_number, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(res_blocks, hidden_layer_number, device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


# def load_model(model_path, hidden_layer_number, device=None):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = ResNet(hidden_layer_number, device)
#
#     if model_path is not None:
#         model.load_state_dict(torch.load(model_path, map_location=device))
#     return model


if __name__ == "__main__":
    import timeit

    game = Othello()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(4, 64, device)
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
