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
            action_probs = self.mcts.predict_best_move(game, all_moves_prob=True)

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


def gen_azero_model(file, hidden_layer=64, res_blocks=4):
    time_limit = math.inf
    iter_limit = iter_depth  # math.inf
    verbose = 0  # 0 means no logging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    m = ResNet(res_blocks, hidden_layer, device)
    # print(f'dir: {os.getcwd()}')
    model_location = file

    m.load_state_dict(torch.load(model_location, map_location=device))
    m.eval()

    return MCTS(f'alpha-mcts - {file}',
                m,
                max_time=time_limit,
                max_iter=iter_limit,
                verbose=verbose)



def model_generator(file_location, model_idxs, hidden_layer=64, res_blocks=4):
    for i in model_idxs:
        model_location = f'{file_location}/model_{i}.pt'
        try:
            model = gen_azero_model(model_location, hidden_layer=hidden_layer, res_blocks=res_blocks)
        except: 
            break
        yield model


def model_generator_all(file_location, hidden_layer=64, res_blocks=4):
    cwd = os.getcwd()
    d = os.path.join(cwd, file_location)
    file_names = [f for f in os.listdir(d)
                  if os.path.isfile(os.path.join(d, f)) and f.startswith('model')]
    
    # In the meantime if new models were created, also detect them
    previous_contents = set()
    
    while True:
        current_contents = set(os.listdir(d))
        new_files = current_contents - previous_contents
        if new_files:
            for f in sorted(list(new_files)):
                    if os.path.isfile(os.path.join(d, f)) and f.startswith('model'):                        
                        model_location = f'{file_location}/{f}'
                        model = gen_azero_model(model_location, hidden_layer=hidden_layer, res_blocks=res_blocks)
                        yield model
        else:
            break
                
        previous_contents = current_contents    


def multi_folder_load_models(folder_hiddenlayer_resblock):
    for folder, layer_number, res_blocks in folder_hiddenlayer:
        print(f'\n++++++++++++++++ TESTED FOLDER - {folder}++++++++++++++++\n')
        yield from model_generator_all(folder, layer_number, res_blocks=res_blocks)
        print()


def multi_folder_load_some_models(folder_hiddenlayer_resblock_model_idxs):
    for folder, layer_number, res_blocks, model_idxs in folder_hiddenlayer_resblock_model_idxs:
        print(f'\n++++++++++++++++ TESTED FOLDER - {folder}++++++++++++++++\n')
        yield from model_generator(folder, model_idxs,layer_number, res_blocks=res_blocks)
        print()



if __name__ != "__main__":
    iter_depth = 50
    print(f'mcts iter depth: {iter_depth}')  
    model_location = f'alpha-zero/low_mcts_iter_training4_128layer/model_11.pt'
    # model_location = f'alpha-zero/low_mcts_iter_training3/model_99.pt'
    # model_location = f'alpha-zero/low_mcts_iter_training/model_66.pt'
    # model_location = f'alpha-zero/model_34_n20+n01.pt'
    mcts_model = None #gen_azero_model(model_location, 128)
          
    # model_location = 'alpha-zero/low_mcts_iter_training4_128layer' # [99, 40, 24, 25, 21, 20, 13, 12, 11, 9, 7] 11 best
    #model_location = 'alpha-zero/low_mcts_iter_training4_128layer_v2'
    #model_location = f'alpha-zero/low_mcts_iter_training4_128layer_v3'
    
    
    hid_layer = 128
    res_blocks = 20
    folder_hiddenlayer = [(f'alpha-zero/res20layer128v9', 128, res_blocks, range(104, 800, 5))]    
    many_models = multi_folder_load_some_models(folder_hiddenlayer)
    
    #folder_hiddenlayer = [(f'alpha-zero/low_mcts_iter_training4_128layer_v{i}', hid_layer, [11, 12, 13, 14, 15]) for i in [9, 10]]
    #folder_hiddenlayer =[#('alpha-zero/low_mcts_iter_training4_128layer_v15', 128, [24]),
                         #('alpha-zero/low_mcts_iter_training4_128layer_v17', 128, [38]),
                         #('alpha-zero/low_mcts_iter_training4_128layer_v18', 128, [23, 53]),
     #                    ('alpha-zero/low_mcts_iter_training4_128layer_v19', 128, [19])]
    #many_models = multi_folder_load_some_models(folder_hiddenlayer) 

def move_to_testing():
    game = Othello()
    game.play_move((2, 4))
    game.play_move((2, 3))
    # print(game)
    encoded_state = game.get_encoded_state()

    tensor_state = torch.tensor(encoded_state).unsqueeze(0)
    model = ResNet(4, 64, torch.device('cpu'))
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(4, 64, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    params = {
        'num_iterations': 200,
        'num_self_play_iterations': 200,
        'num_epochs': 20,
        'batch_size': 64,
        'temp': 1.1
    }
    mcts_params = {
        'uct_exploration_const': 2,
        'max_iter': 500,
        # these are flexible dirichlet epsilon for noise
        # favor exploration more in the beginning
        'initial_alpha': 0.4,
        'final_alpha': 0.1,
        'decay_steps': 30
    }
    azero = AlphaZero(model, optimizer, params, mcts_params)

    execution_time = timeit.timeit(azero.learn, number=1)
    print(f"Execution time: {execution_time} seconds")

    # move_to_testing()
