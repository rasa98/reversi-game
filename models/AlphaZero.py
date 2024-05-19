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
    def __init__(self, game, num_resBlocks, num_hidden):
        super().__init__()
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
            player = game.player_turn
            encoded_perspective_state = Othello.get_encoded_state(game.board, player)
            action_probs = self.mcts.predict_best_move(game, all_moves_prob=True)

            data.append((encoded_perspective_state, action_probs, player))

            action = np.random.choice(ALL_FIELDS_SIZE, p=action_probs)

            game.play_move(Othello.get_decoded_field(action))
            if game.is_game_over():
                data_to_return = []
                winner = game.get_winner()
                for state, action_probs, player in data:
                    value = 0
                    if winner == player:
                        value = 1
                    elif winner != 0: #  if its not a draw
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

            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.params['num_iterations']):
            memory = []

            self.model.eval()
            for _ in trange(self.params['num_self_play_iterations']):
                memory += self.self_play()

            self.model.train()
            for _ in trange(self.params['num_epochs']):
                self.train(memory)

            folder = 'alpha-zero'
            os.makedirs(folder, exist_ok=True)

            torch.save(self.model.state_dict(), f"{folder}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"{folder}/optimizer_{iteration}.pt")


time_limit = 1
iter_limit = 1500  # math.inf
verbose = 1  # 0 means no logging

m = ResNet(Othello, 4, 64)
m.eval()

mcts_model = MCTS(f'alpha-mcts {time_limit}s',
                  m,
                  max_time=time_limit,
                  max_iter=iter_limit,
                  verbose=verbose)


def move_to_testing():
    game = Othello()
    game.play_move((2, 4))
    game.play_move((2, 3))
    # print(game)
    encoded_state = Othello.get_encoded_state(game.board, 1)

    tensor_state = torch.tensor(encoded_state).unsqueeze(0)
    model = ResNet(game, 4, 64)
    policy, value = model(tensor_state)
    value = value.item()
    policy = (torch.softmax(policy, dim=1).squeeze(0).detach()
              .cpu()
              .numpy())

    # print(value, policy)
    res = game.valid_moves_encoded() * policy
    print(res / np.sum(res))


if __name__ == "__main__":
    import timeit

    game = Othello()
    model = ResNet(game, 4, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = {
        'num_iterations': 5,
        'num_self_play_iterations': 32,
        'num_epochs': 10,
        'batch_size': 8
    }
    mcts_params = {
        'uct_exploration_const': 2,
        'max_iter': 50,
    }
    azero = AlphaZero(model, optimizer, params, mcts_params)

    execution_time = timeit.timeit(azero.learn, number=1)
    print(f"Execution time: {execution_time} seconds")


    # move_to_testing()
