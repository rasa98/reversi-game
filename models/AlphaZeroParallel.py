import numpy as np
import math
import json
import random
import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

# for teseting purposes
import sys
import os


#source_dir = os.path.abspath(os.path.join(os.getcwd(), '../'))
#sys.path.append(source_dir)
# ---------------------
from game_logic import Othello
from models.mcts_alpha_parallel import MCTS

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
        self.model_output = params['model_output']
        self.save_every = params['model_save_x_iter']

        self.mcts = MCTS("alpha-mcts", model, **mcts_params)

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
        
        train_policy_loss /= len(data) // self.params['batch_size']
        train_value_loss /= len(data) // self.params['batch_size']
        scale = 1000
        print(f"Epoch {epoch + 1}/{params['num_epochs']} - \n"
                  f"Train Policy Loss: {scale * train_policy_loss}, "
                  f"Train Value Loss: {scale * train_value_loss}")

        val_policy_loss, val_value_loss = self.validate_loss(val_data)
        print(#f"Epoch {epoch + 1}/{params['num_epochs']} - "
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
            print(f'Learning Iteration - {iteration}')
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
            
            if iteration > self.params['model_save_after_x_iter'] and (iteration + 1) % self.save_every == 0:
                torch.save(self.model.state_dict(), f"{folder}/model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"{folder}/optimizer_{iteration}.pt")

            self.model.iterations_trained += 1

def load_model_and_optimizer(model, optimizer, model_state_path, optimizer_state_path, device):
    model.load_state_dict(torch.load(model_state_path))
    model.to(device)
    optimizer.load_state_dict(torch.load(optimizer_state_path))

class SPG:
    def __init__(self):
        self.game = Othello()
        self.data = []
        self.root = None
        # self.node = None


if __name__ == "__main__":    
    params = {
        'res_blocks': 20,
        'hidden_layer': 128,
        'lr': 0.0003,
        'weight_decay': 0.08,
        'num_iterations': 500,
        'num_self_play_iterations': 50,
        'num_epochs': 5,
        'batch_size': 128,
        'temp': 1.03,
        'num_parallel_games': 50,
        'model_save_x_iter': 5,
        'model_save_after_x_iter': 0,
        'model_output': 'alpha-zero/res20layer128vF'               
    }
    mcts_params = {
        'uct_exploration_const': 2,
        'max_iter': 125,        
        # these are flexible dirichlet epsilon for noise
        # favor exploration more in the beginning
        'dirichlet_epsilon': 0.05,
        'initial_alpha': 0.4,
        'final_alpha': 0.05,
        'decay_steps': 100
    }  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # Initialize model and optimizer
    model = ResNet(params['res_blocks'], params['hidden_layer'], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    #optimizer = torch.optim.SGD(model.parameters(), params['lr'], weight_decay=params['weight_decay'])
    
    print(F'CURRENT WORKING DIR - {os.getcwd()}')
    if len(sys.argv) > 2:
        model_state_path = sys.argv[1]
        optimizer_state_path = sys.argv[2]        
        print(f'----------STARTING MODEL - {model_state_path}----------')
        # Load model and optimizer states
        load_model_and_optimizer(model, optimizer, model_state_path, optimizer_state_path, device)          
    
    print(f'\nparams \n{json.dumps(params, indent=4)}')    
    print(f'\nmcts_maprams \n{json.dumps(mcts_params, indent=4)}\n')

    azero = AlphaZero(model, optimizer, params, mcts_params)   
    azero.learn()