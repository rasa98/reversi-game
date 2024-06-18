import numpy as np
import math
import json
import random
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import trange
from torch.optim.lr_scheduler import StepLR

# for teseting purposes
import sys
import copy
import os
# ---------------------
from game_logic import Othello
from models.mcts_alpha_parallel import MCTS

from models.ppo_masked_model import load_model_new
from bench_agent import bench_both_sides
from models.montecarlo_alphazero_version import MCTS as MCTS1
from models.model_interface import ai_random
from models.AlphaZeroModel import load_azero_model

GAME_ROW_COUNT = 8
GAME_COLUMN_COUNT = 8
ALL_FIELDS_SIZE = GAME_ROW_COUNT * GAME_COLUMN_COUNT


# class ResNet(nn.Module):
#     def __init__(self, num_resBlocks, num_hidden, device):
#         super().__init__()
#         self.device = device
#         self.iterations_trained = 0
#
#         self.startBlock = nn.Sequential(
#             nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
#             nn.BatchNorm2d(num_hidden),
#             nn.ReLU()
#         )
#
#         self.backBone = nn.ModuleList(
#             [ResBlock(num_hidden) for i in range(num_resBlocks)]
#         )
#
#         self.policyHead = nn.Sequential(
#             nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(32 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, ALL_FIELDS_SIZE)
#         )
#
#         self.valueHead = nn.Sequential(
#             nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
#             nn.BatchNorm2d(3),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(3 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, 1),
#             nn.Tanh()
#         )
#
#         self.to(device)
#
#     def forward(self, x):
#         x = self.startBlock(x)
#         for resBlock in self.backBone:
#             x = resBlock(x)
#         policy = self.policyHead(x)
#         value = self.valueHead(x)
#         return policy, value
#
#
# class ResBlock(nn.Module):
#     def __init__(self, num_hidden):
#         super().__init__()
#         self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(num_hidden)
#         self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(num_hidden)
#
#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x += residual
#         x = F.relu(x)
#         return x

class ResNet(nn.Module):
    def __init__(self, num_hidden, device):
        super().__init__()
        self.device = device
        self.iterations_trained = 0

        self.device = device

        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3)  # Dropout layer
        )
        
        self.sharedConv = nn.Sequential(
            nn.Conv2d(num_hidden, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3)  # Dropout layer

        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, ALL_FIELDS_SIZE)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * GAME_ROW_COUNT * GAME_COLUMN_COUNT, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        shared_out = self.sharedConv(x)
        policy = self.policyHead(shared_out)
        value = self.valueHead(shared_out)
        return policy, value


class AlphaZero:
    def __init__(self, model, optimizer, scheduler, params, mcts_params):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.model_output = params['model_output']

        self.best_model = None
        self.best_models_optimizer = None
        self.test_agent = self.load_test_agent()
        self.mcts = MCTS(model, **mcts_params)
        self.model_iteration = 0

        self.model_subsequent_fail = 0
        self.max_fail_times = params['model_subsequent_fail']

        self.scheduler = scheduler
        self.copy_model()

    @staticmethod
    def load_test_agent():
        ppo_18_big_rollouts = (
            load_model_new('strong 17 ppo',
                           'scripts/rl/output/v3v3/history_0017'))

        return ppo_18_big_rollouts

    def copy_model(self):
        self.best_model = copy.deepcopy(self.model)
        self.best_models_optimizer = torch.optim.Adam(self.best_model.parameters())
        self.best_models_optimizer.load_state_dict(self.optimizer.state_dict())

    @staticmethod
    def bench_agents(a1, a2, times=10):
        a1_wins, a2_wins = bench_both_sides(
            a1,
            a2,
            times=times,
            timed=True,
            verbose=1)
        a1_win_rate = a1_wins / (2 * times)
        return a1_wins, a2_wins, a1_win_rate

    def save_if_passes_bench(self, folder, iteration):
        current_agent = load_azero_agent("current", model=self.model, params=params)
        best_agent = load_azero_agent('best yet', model=self.best_model, params=params)
        test_agent = self.test_agent

        print(f'\n×××××  benchmarking model after training  ×××××')
        _, _, a1_winrate = self.bench_agents(current_agent, ai_random)
        if a1_winrate < 0.8:
            return False

        if self.model_iteration > 4:
            current_wins, test_wins, _ = self.bench_agents(current_agent, test_agent)
            if current_wins <= test_wins:
                return False

            _, _, a1_winrate = self.bench_agents(current_agent, best_agent)
            if a1_winrate < 0.6:
                return False
        torch.save(self.model.state_dict(), f"{folder}/model_{iteration}.pt")
        torch.save(self.optimizer.state_dict(), f"{folder}/optimizer_{iteration}.pt")
        self.copy_model()
        self.model_iteration += 1
        return True

    def self_play(self):
        data_to_return = []
        # game = Othello()
        spGames = [SPG() for _ in range(self.params['num_parallel_games'])]

        # while True:
        while len(spGames) > 0:
            action_probs = self.mcts.predict_best_move(spGames)
            # print(f'before - {action_probs.shape}')
            for i in range(len(spGames) - 1, -1, -1):
                # print(f'after 0 - {action_probs.shape}')
                spg = spGames[i]
                encoded_perspective_state = spg.game.get_encoded_state()
                game = spg.game

                # print(f'after 1 - {action_probs.shape}')
                spg.data.append((encoded_perspective_state,
                                 action_probs[i],
                                 game.player_turn))
                # print(f'after 2 - {action_probs.shape}')
                temp_action_probs = action_probs[i] ** (1 / self.params['temp'])
                temp_action_probs /= np.sum(temp_action_probs)
                # print(f'after 4 - {action_probs.shape}')

                action = np.random.choice(ALL_FIELDS_SIZE, p=temp_action_probs)

                # print(f'after 5 - {action_probs.shape}')
                game.play_move(Othello.get_decoded_field(action))
                if game.is_game_over():
                    winner = game.get_winner()
                    for state, probs, player in spg.data:
                        value = 0
                        if winner == player:
                            value = 1
                        elif winner != 0:  # if its not a draw
                            value = -1

                        data_to_return.append((
                            state,
                            probs,
                            value
                        ))
                    del spGames[i]
                    # print(f'after 6 - {action_probs.shape}')
                # print(f'after 7 - {action_probs.shape}')
        return data_to_return

    def train(self, data, val_data, epoch):
        random.shuffle(data)
        train_policy_loss = 0
        train_value_loss = 0
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

            train_policy_loss += policy_loss
            train_value_loss += value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.scheduler.step()

        train_policy_loss /= len(data) // self.params['batch_size']
        train_value_loss /= len(data) // self.params['batch_size']
        scale = 1000
        print(f"Epoch {epoch + 1}/{params['num_epochs']} - \n"
              f"Train Policy Loss: {scale * train_policy_loss}, "
              f"Train Value Loss: {scale * train_value_loss}"
              f", lr: {self.scheduler.get_last_lr()[0]}")

        val_policy_loss, val_value_loss = self.validate_loss(val_data)
        print(  # f"Epoch {epoch + 1}/{params['num_epochs']} - "
            f"Valid Policy Loss: {scale * val_policy_loss}, "
            f"Valid Value Loss: {scale * val_value_loss}", flush=True)

    def validate_loss(self, data):
        self.model.eval()
        policy_loss = 0
        value_loss = 0
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

            policy_loss += policy_loss.item()
            value_loss += value_loss.item()
        policy_loss /= len(data) // self.params['batch_size']
        value_loss /= len(data) // self.params['batch_size']
        self.model.train()
        return policy_loss, value_loss

    def learn(self):
        folder = self.model_output
        if os.path.isdir(folder) and len(os.listdir(folder)) > 0:
            raise Exception("Model output already exists. You probably need to update it!!!")
        os.makedirs(folder, exist_ok=True)

        for iteration in range(self.params['num_iterations']):            
            memory = []

            self.model.eval()
            for _ in trange(self.params['num_self_play_iterations'] // self.params['num_parallel_games']):
                memory += self.self_play()

            self.model.train()
            random.shuffle(memory)
            separation_idx = int(0.2 * len(memory))
            val_data = memory[0: separation_idx]
            train_data = memory[separation_idx:]
            for epoch in range(self.params['num_epochs']):
                self.train(train_data, val_data, epoch)

            if not self.save_if_passes_bench(folder, iteration):
                self.model_subsequent_fail += 1
                if self.model_subsequent_fail > self.max_fail_times:
                    print(f'FAILED TO SATISFY BENCHMARKS {self.max_fail_times} times in a row. Reseting to earlier best model...')

                    self.model = self.best_model
                    self.optimizer = self.best_models_optimizer
                    self.model_subsequent_fail = 0
            else:
                self.model_subsequent_fail = 0
                self.model.iterations_trained += 1


def load_model_and_optimizer(params, model_state_path, optimizer_state_path, device):
    model = ResNet(params['hidden_layer'], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    if model_state_path is not None:
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))

        for param_group in optimizer.param_groups:
            param_group['lr'] = params['lr']
            param_group['weight_decay'] = params['weight_decay']

    scheduler = StepLR(optimizer, step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma'])

    return model, optimizer, scheduler


class SPG:
    def __init__(self):
        self.game = Othello()
        self.data = []
        self.root = None
        # self.node = None


if __name__ == "__main__":
    if os.environ['USER'] == 'rasa':
        source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
        sys.path.append(source_dir)
        os.chdir('../')

    params = {
        #'res_blocks': 4,
        'hidden_layer': 128,
        'lr': 7e-5,
        'weight_decay': 1e-4,
        'num_iterations': 50,
        'num_self_play_iterations': 800,
        'num_epochs': 6,
        'batch_size': 128,
        'temp': 1.1,
        'num_parallel_games': 50,
        'model_subsequent_fail': 5,
        'scheduler_step_size': 12, 
        'scheduler_gamma':0.97,
        'model_output': 'models_output/alpha-zero/FINAL/layer128-v2/'
    }
    mcts_params = {
        'uct_exploration_const': 1.4,
        'max_iter': 50,
        # these are flexible dirichlet epsilon for noise
        # favor exploration more in the beginning
        'dirichlet_epsilon': 0.25,
        'initial_alpha': 0.4,
        'final_alpha': 0.15,
        'decay_steps': 30
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_state_path = None
    optimizer_state_path = None
    if len(sys.argv) > 2:
        model_state_path = sys.argv[1]
        optimizer_state_path = sys.argv[2]
    print(f'----------STARTING MODEL - {model_state_path}----------')

    model, optimizer, scheduler = load_model_and_optimizer(params,
                                                           model_state_path,
                                                           optimizer_state_path,
                                                           device)


    print(f'\nparams \n{json.dumps(params, indent=4)}')
    print(f'\nmcts_maprams \n{json.dumps(mcts_params, indent=4)}\n')

    azero = AlphaZero(model, optimizer, scheduler, params, mcts_params)
    azero.learn()
