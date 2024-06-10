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

# TODO: dupliran kod iz Alphazero....
GAME_ROW_COUNT = 8
GAME_COLUMN_COUNT = 8
ALL_FIELDS_SIZE = GAME_ROW_COUNT * GAME_COLUMN_COUNT


class Node:
    def __init__(self, game_copy, move, parent_node=None, visited=0, prior=0):
        self.game = game_copy
        self.visited = visited
        self.move = move
        self.children = []
        self.value = 0
        self.prior = prior
        # self.move_to_child = {}
        self.parent = parent_node
        #self.valid_moves = list(game_copy.valid_moves_to_reverse)
        self.is_final_state = self.game.is_game_over()

    # def get_all_next_move_counter(self):
    #     """for debug"""
    #     return Counter({move: child.visited for move, child in self.move_to_child.items()})

    # def explored_OLD(self):
    #     return len(self.valid_moves) == 0

    def explored(self):
        return len(self.children) > 0

    def explore_all_children(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                move = Othello.get_decoded_field(action)
                game_copy = self.game.get_snapshot()
                game_copy.play_move(move)
                child_node = Node(game_copy, move, parent_node=self, prior=prob)
                self.children.append(child_node)

    def select_highest_ucb_child(self, c):
        max_child = max(self.children, key=lambda ch: ch.get_uct(c, self.visited))
        return max_child

    def get_uct(self, c, parent_visits):
        if self.visited == 0:
            q_value = 0
        else:
            #q_value = self.value / self.visited
            # q_value = 1 - (self.value / self.visited + 1) / 2
            avg_value = self.value / self.visited
            q_value = (avg_value + 1) / 2
        exploration_term = c * (math.sqrt(parent_visits) / (self.visited + 1))
        explo_biased = exploration_term * (self.prior + 0.5)

        return q_value + explo_biased

    def simulate_game(self):
        game_copy = self.game.get_snapshot()
        winner = ai_vs_ai_cli(ai_random, ai_random, game_copy)  # 1, 2, or 0
        return winner


class MCTS():

    def __init__(self, name, model, max_iter=math.inf, max_time=math.inf,
                 uct_exploration_const=2.0, verbose=0, dirichlet_epsilon=0.2,
                 initial_alpha=0.4, final_alpha=0.1, decay_steps=50):
        # super().__init__(name)
        # self.root = None
        self.model = model

        self.last_cycle_iteration = 0
        self.last_cycle_time = 0

        self.max_iter = max_iter
        self.max_time = max_time
        self.verbose = verbose

        self.dirichlet_epsilon = dirichlet_epsilon
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.decay_steps = decay_steps

        self.uct_exploration_const = uct_exploration_const

    def iter_per_cycle(self):
        if self.last_cycle_time != 0:
            return int(self.last_cycle_iteration / self.last_cycle_time)
        return -1

    def set_root_new(self, spgs):
        """create new root Node."""
        games_copy = [spg.game.get_snapshot() for spg in spgs]
        policies, _ = self.run_model(games_copy, add_noise=True)
        # TODO get back and finish

        for i, spg in enumerate(spgs):
            spg.root = Node(games_copy[i], None, visited=1)
            policy = policies[i]
            spg.root.explore_all_children(policy)

    def predict_best_move(self, spgs):
        self.set_root_new(spgs)
        self.mcts_search(spgs)
        # print(f'\nafter simulating: {dict(self.root.get_all_next_move_counter())}')
        gc.collect()

        action_probs = np.zeros((len(spgs), ALL_FIELDS_SIZE))

        for i, spg in enumerate(spgs):
            for child in spg.root.children:
                move = child.move
                encoded_move = Othello.get_encoded_field(move)
                action_probs[i][encoded_move] = child.visited
            action_probs[i] /= np.sum(action_probs[i])

        return action_probs

    def best_move_child_items(self):
        best_items = []
        max_value = -math.inf

        for node in self.root.children:
            move = node.move
            if (current_value := node.visited) > max_value:
                max_value, best_items = current_value, [(move, node)]
            elif current_value == max_value:
                best_items.append((move, node))

        return best_itemsselect_expansion_sim

    def best_moves(self):
        items = self.best_move_child_items()  # [(move, child),..]
        return [el[0] for el in items]  # [move1, move2, ...]

    def select_expansion_sim(self, spgs):
        nodes = []
        values = []

        games = []
        nodes_to_expand = []
        for spg in spgs:
            node: Node = spg.root
            value = 0
            while True:
                if node.is_final_state:
                    winner = node.game.get_winner()
                    if node.game.last_turn == winner:
                        value = 1
                    elif winner != 0:
                        value = -1  # else it stays 0
                    nodes.append(node)
                    values.append(value)
                    break  # return node
                elif not node.explored():
                    # policy, value = self.run_model(node.game)
                    # value = value.item()
                    #
                    # node.explore_all_children(policy)  # return node.explore_new_child()
                    # spg.node = node
                    nodes_to_expand.append(node)
                    games.append(node.game)
                    break
                else:
                    node = node.select_highest_ucb_child(self.uct_exploration_const)

        if len(games) > 0:
            policies, vals = self.run_model(games)
            for i, node in enumerate(nodes_to_expand):
                node.explore_all_children(policies[i])
                nodes.append(node)
                values.append(vals[i][0])

        return nodes, values  # returns (node , winner)

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

    @torch.no_grad()
    def run_model(self, games, add_noise=False):
        states = np.stack([game.get_encoded_state() for game in games])

        policy, value = self.model(
            torch.tensor(states, device=self.model.device, dtype=torch.float32)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        value = value.cpu().numpy()

        if add_noise:
            policy = policy * (1 - self.dirichlet_epsilon) + self.dirichlet_epsilon \
                     * np.random.dirichlet([self.get_dirichlet_alpha()] * ALL_FIELDS_SIZE,
                                           size=policy.shape[0])

        for i, game in enumerate(games):
            valid_moves = game.valid_moves_encoded()  # TODO mozda spg.root.game ???
            policy[i] *= valid_moves
            policy[i] /= np.sum(policy[i])

        return policy, value

    def get_dirichlet_alpha(self):
        # Linear decay of alpha
        training_step = self.model.iterations_trained
        if training_step >= self.decay_steps:
            return self.final_alpha
        else:
            return self.initial_alpha - (self.initial_alpha - self.final_alpha) * (training_step / self.decay_steps)

    def mcts_iter(self, spgs):
        nodes, values = self.select_expansion_sim(spgs)
        for node, value in zip(nodes, values):
            self.backprop(node, value)

    def mcts_search(self, spgs):
        if self.max_time == math.inf and self.max_iter == math.inf:
            raise ValueError("At least one of max_time or max_iter must be specified.")

        start_time = time.perf_counter()

        iterations = 1  # root is always called in advance to add some noise
        while True:
            # Perform MCTS steps: selection, expansion, simulation, backpropagation
            self.mcts_iter(spgs)
            iterations += 1

            print(f'iteration {iterations} done!')
            if iterations > 45 and spgs[0].game.turn > 45: # TODO delete DEBUG
                pass

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
