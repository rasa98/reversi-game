import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange

# for teseting purposes
import sys
import os

source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
sys.path.append(source_dir)
# ---------------------
from game_logic import Othello
from .montecarlo_alphazero_version import MCTS

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
    def __init__(self, model, optimizer, params):
        self.model = model
        self.optimizer = optimizer
        self.params = params

        self.mcts = MCTS("alpha-mcts", model, **params)

    def self_play(self):
        memory = []
        player = 1
        game = Othello()
        state = game.board

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, data):
        pass

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for _ in trange(self.args['num_self_play_iterations']):
                memory += self.self_play()

            self.model.train()
            for _ in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


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
    game = Othello()
    model = ResNet(game, 4, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = {
        'uct_exploration_const': 2,
        'max_iter': 60,
        'num_iterations': 3,
        'num_self_play_iterations': 10,
        'num_epochs': 4

    }
    azero = AlphaZero(game, model, optimizer, params)
    azero.learn()

    # move_to_testing()
