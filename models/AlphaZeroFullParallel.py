import numpy as np
import torch.multiprocessing as mp
import math
import json
import random
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from tqdm import trange

# for teseting purposes
import sys
import copy
import os

# import logging
# logging.getLogger('tensorflow').disabled = True
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if os.environ['USER'] == 'rasa':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
    sys.path.append(source_dir)
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
#
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
    def __init__(self, model, optimizer, scheduler, params, mcts_params):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params = params
        self.mcts_params = mcts_params
        self.model_output = params['model_output']

        self.best_model = None
        self.best_models_optimizer = None
        self.test_agent = self.load_test_agent()
        self.mcts = MCTS(model, **mcts_params)

        self.model_subsequent_fail = 0
        self.max_fail_times = params['max_fail_times']

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
    def bench_agents(a1, a2, times=10, det=True):
        a1.set_deterministic(det)
        a2.set_deterministic(det)

        a1_wins, a2_wins = bench_both_sides(
            a1,
            a2,
            times=times,
            timed=True,
            verbose=1)
        a1_win_rate = a1_wins / (2 * times)
        return a1_wins, a2_wins, a1_win_rate   

    def save_if_passes_bench(self, folder, iteration):
        params = dict(self.mcts_params)
        params['max_iter'] = 30
        params['dirichlet_epsilon'] = 0.2  # for some variability when simulating
        params['uct_exploration_const'] = 1.41
        params['decay_steps'] = -1  # so that alpha is set to final alpha for both
        params['final_alpha'] = 0.1        


        assert params is not self.mcts_params, \
            'You changed azero obj field mcts_params!!'

        current_agent = load_azero_model("current", model=self.model, params=params)
        best_agent = load_azero_model('best yet', model=self.best_model, params=params)
        test_agent = self.test_agent

        print(f'\n×××××  benchmarking model after training  ×××××')
        # _, _, a1_winrate = self.bench_agents(current_agent, ai_random)
        # if a1_winrate < 0.8:
        #     return False
        #
        # if self.model_iteration > -1:
        #     current_wins, test_wins, a1_winrate = self.bench_agents(current_agent, test_agent, det=False)
        #     if a1_winrate < 0.8:
        #         return False
        #
        #     _, _, a1_winrate = self.bench_agents(current_agent, best_agent, det=False)
        #     if a1_winrate < 0.65:
        #         return False
        _, _, a1_winrate = self.bench_agents(current_agent, best_agent, times=20, det=True)
        if a1_winrate < 0.65:
            return False
        print('++++ +++++++++ ++++New model Save!!!++++ +++++++++ ++++')
        torch.save(self.model.state_dict(), f"{folder}/model_{iteration}.pt")
        torch.save(self.optimizer.state_dict(), f"{folder}/optimizer_{iteration}.pt")
        self.copy_model()
        return True

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

        self.model.share_memory()

        for iteration in range(self.params['num_iterations']):
            memory = []

            self.model.eval()

            times = ((self.params['num_self_play_iterations'] // self.params['num_parallel_games']) // num_cores)
            for _ in trange(1):  # just to time it
                unflattened_memory = pool.starmap(parallel_fun,
                                                  [(rank, times, self.params, self.model, self.mcts_params) for rank in
                                                   range(num_cores)])

            for data in unflattened_memory:
                memory.extend(data)

            print(flush=True)

            self.model.train()
            random.shuffle(memory)
            separation_idx = int(0.2 * len(memory))
            val_data = memory[0: separation_idx]
            train_data = memory[separation_idx:]
            for epoch in range(self.params['num_epochs']):
                self.train(train_data, val_data, epoch)

            if not self.save_if_passes_bench(folder, iteration):
                self.model_subsequent_fail += 1
                if self.model_subsequent_fail >= self.max_fail_times:
                    print(
                        f'FAILED TO SATISFY BENCHMARKS {self.max_fail_times} times in a row. Reseting to earlier best model...')
                    self.model = self.best_model
                    self.optimizer = self.best_models_optimizer
                    self.model_subsequent_fail = 0
            else:
                self.model_subsequent_fail = 0
                self.model.iterations_trained += 1  # ovo ne treba van if/else??? nije tolko bitno,
                                                    # za dirichlet alpha se koristi.


def load_model_and_optimizer(params, model_state_path, optimizer_state_path, device):
    model = ResNet(params['res_blocks'], params['hidden_layer'], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    if model_state_path is not None:
        model.load_state_dict(torch.load(model_state_path, map_location=device))
        optimizer.load_state_dict(torch.load(optimizer_state_path))

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


lock = None


def init(_lock):
    global lock
    lock = _lock


def parallel_fun(rank, times, params, model, mcts_params):
    random_data = os.urandom(8)
    seed = int.from_bytes(random_data, byteorder="big")
    seed += rank
    seed = seed % (2 ** 32 - 1)

    random.seed(seed)
    np.random.seed(seed)

    res = []

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if str(model.device) not in {'cuda'}:
        raise Exception('Not running on CUDA !!!!')

    mcts = MCTS(model, seed=seed, lock=lock, **mcts_params)

    for _ in range(times):
        res += self_play_function(params, mcts)
    return res


def get_temp_value(training_step, params):
    # Linear decay of temp
    temp_decay = params['temp_decay_steps']
    initial_temp = params['initial_temp']
    final_temp = params['final_temp']
    
    if training_step >= temp_decay:
        return final_temp
    else:
        return initial_temp - (initial_temp - final_temp) * (training_step / temp_decay)



def self_play_function(params, mcts):
    data_to_return = []
    spGames = [SPG() for _ in range(params['num_parallel_games'])]

    # if str(mcts.model.device) not in {'cuda', 'cuda:0', 'cuda:1'}:
    #    raise Exception('Not running on CUDA !!!!')

    while len(spGames) > 0:
        action_probs = mcts.predict_best_move(spGames)
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
            temp = get_temp_value(self.model.iterations_trained, params)
            temp_action_probs = action_probs[i] ** (1 / temp)
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


if __name__ == "__main__":
    if os.environ['USER'] == 'rasa':
        print('running on local node')
        os.chdir('../')
        num_cores = 2
    else:
        num_cores = int(os.environ['SLURM_CPUS_ON_NODE']) // 2

    print(f'number of cores used for pool: {num_cores}')

    params = {
        'res_blocks': 8,
        'hidden_layer': 128,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'num_iterations': 30,
        'num_self_play_iterations': 1 * 64 * 16,
        'num_epochs': 5,
        'batch_size': 128,
        'initial_temp': 1.75,
        'final_temp': 0.6,
        'temp_decay_steps': 16,
        'num_parallel_games': 64,
        'max_fail_times': 2,
        'scheduler_step_size': 5, 
        'scheduler_gamma':0.8,
        'model_output': 'models_output/alpha-zero/FINAL2/res8layer128-v1/'
    }
    mcts_params = {
        'uct_exploration_const': 1.72,
        'max_iter': 100,#70,
        # these are flexible dirichlet epsilon for noise
        # favor exploration more in the beginning
        'dirichlet_epsilon': 0.2,
        'initial_alpha': 0.6,
        'final_alpha': 0.1,
        'decay_steps': 8
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

    mp.set_start_method('forkserver')
    lock = None #mp.Lock()

    with mp.Pool(processes=num_cores, initializer=init, initargs=(lock,)) as pool:
        azero.learn()
