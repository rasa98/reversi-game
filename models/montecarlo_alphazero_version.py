import math
import numpy
import numpy as np
import time
import gc

import torch
from game_logic import Othello
from game_modes import ai_vs_ai_cli
from models.model_interface import ai_random
from collections import Counter

from .model_interface import ModelInterface
from .AlphaZero import ResNet


class Node:

    def __init__(self, game_copy, parent_node=None, prior=0):
        self.game = game_copy
        self.visited = 0
        self.value = 0
        self.prior = prior
        self.move_to_child = {}
        self.parent = parent_node
        self.valid_moves = list(game_copy.valid_moves_to_reverse)
        self.is_final_state = self.game.is_game_over()

    def get_all_next_move_counter(self):
        """for debug"""
        return Counter({move: child.visited for move, child in self.move_to_child.items()})

    def explored_OLD(self):
        return len(self.valid_moves) == 0

    def explored(self):
        return len(self.move_to_child) > 0

    def explore_all_children(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                move = Othello.get_decoded_field(action)
                game_copy = self.game.get_snapshot()
                game_copy.play_move(move)
                child_node = Node(game_copy, parent_node=self, prior=prob)
                self.move_to_child[move] = child_node

    def select_highest_ucb_child(self, c):
        max_child = max(self.move_to_child.values(), key=lambda ch: ch.get_uct(c, self.visited))
        return max_child

    def get_uct(self, c, parent_visits):
        if self.visited == 0:
            q_value = 0
        else:
            # q_value = self.value / self.visited
            q_value = 1 - (self.value / self.visited + 1) / 2
        exploration_term = c * (math.sqrt(parent_visits) / (self.visited + 1)) * self.prior
        return q_value + exploration_term

    def simulate_game(self):
        game_copy = self.game.get_snapshot()
        winner = ai_vs_ai_cli(ai_random, ai_random, game_copy)  # 1, 2, or 0
        return winner


class MCTS(ModelInterface):

    def __init__(self, name, model, max_iter=math.inf, max_time=math.inf, uct_exploration_const=2.0, verbose=0):
        super().__init__(name)
        self.root = None  # Node(game.get_snapshot())
        self.model = model

        self.last_cycle_iteration = 0
        self.last_cycle_time = 0

        self.max_iter = max_iter
        self.max_time = max_time
        self.verbose = verbose

        self.uct_exploration_const = uct_exploration_const

    def iter_per_cycle(self):
        if self.last_cycle_time != 0:
            return int(self.last_cycle_iteration / self.last_cycle_time)
        return -1

    def set_root_new(self, game):
        """create new root Node."""
        self.root = Node(game.get_snapshot())

    def predict_best_move(self, game: Othello):
        self.set_root_new(game)
        self.mcts_search()
        # print(f'\nafter simulating: {dict(self.root.get_all_next_move_counter())}')
        gc.collect()
        return self.best_moves(), None

    # def best_move_child_item(self):
    #     return max(self.root.move_to_child.items(), key=lambda item: item[1].visited)

    def best_move_child_items(self):
        best_items = []
        max_value = -math.inf

        for move, node in self.root.move_to_child.items():
            if (current_value := node.visited) > max_value:
                max_value, best_items = current_value, [(move, node)]
            elif current_value == max_value:
                best_items.append((move, node))

        return best_items

    def best_moves(self):
        items = self.best_move_child_items()  # [(move, child),..]
        return [el[0] for el in items]  # [move1, move2, ...]

    @torch.no_grad()
    def select_expansion_sim(self):
        node: Node = self.root
        value = 0
        while True:
            if node.is_final_state:
                winner = node.game.get_winner()
                if node.game.last_turn == winner:
                    value = 1
                elif winner != 0:
                    value = -1  # else it stays 0
                break  # return node
            elif not node.explored():
                encoded_state = Othello.get_encoded_state(node.game.board, node.game.player_turn)
                policy, value = self.model(
                    torch.tensor(encoded_state).unsqueeze(0)
                )
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = node.game.valid_moves_encoded()

                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.explore_all_children(policy)  # return node.explore_new_child()
                break
            else:
                node = node.select_highest_ucb_child(self.uct_exploration_const)
        return node, value  # returns (node , winner)

    def backprop(self, node: Node, value: float):
        from_perspective_of = node.game.last_turn
        while node is not None:
            node.visited += 1
            if from_perspective_of == node.game.last_turn:
                node.value += value
            else:
                node.value -= value
            node = node.parent

    @staticmethod
    def is_time_limit_reached(start_time, max_time_sec):
        elapsed_time = time.time() - start_time
        return elapsed_time >= max_time_sec

    def mcts_iter(self):
        node, value = self.select_expansion_sim()
        self.backprop(node, value)

    def mcts_search(self):
        if self.max_time == math.inf and self.max_iter == math.inf:
            raise ValueError("At least one of max_time or max_iter must be specified.")

        start_time = time.perf_counter()
        iterations = 0
        while True:
            # Perform MCTS steps: selection, expansion, simulation, backpropagation
            self.mcts_iter()
            iterations += 1

            # Check termination conditions
            check_time = time.perf_counter()
            elapsed_time = check_time - start_time
            if elapsed_time >= self.max_time or iterations >= self.max_iter:
                break
        self.last_cycle_time = elapsed_time
        self.last_cycle_iteration = iterations

        if self.verbose:
            print(f'game turn: {self.root.game.turn}')
            print(f'iterations - {iterations}, iter per second: {self.iter_per_cycle()}\n')


time_limit = 1
iter_limit = 5000  # math.inf
verbose = 1  # 0 means no logging

m = ResNet(Othello, 4, 64)
m.eval()

mcts_model = MCTS(f'mcts {time_limit}s',
                  m,
                  max_time=time_limit,
                  max_iter=iter_limit,
                  verbose=verbose)