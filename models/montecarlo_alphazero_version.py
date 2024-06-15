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
        # self.valid_moves = list(game_copy.valid_moves_to_reverse)
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

    # def select_highest_ucb_child(self, c):
    #     log_visited = math.log(self.visited)
    #     max_child = max(self.children, key=lambda ch: ch.get_uct(c, log_visited)[0])
    #     mapped_children_DEBUG = list(map(lambda ch: ch.get_uct(c, log_visited)[1], self.children))
    #     return max_child
    #
    # def get_uct(self, c, log_visited):
    #     if self.visited == 0:
    #         q_value = 0
    #     else:
    #         # q_value = self.value / self.visited
    #         avg_value = self.value / self.visited
    #         q_value = avg_value * 4
    #     exploration_term = c * math.sqrt(log_visited / (1 + self.visited))#(math.sqrt(parent_visits) / (self.visited + 1))
    #     # q_value *= (self.prior + 0.5)
    #     exploration_term *= (self.prior + 0.075)
    #     # explo_biased = exploration_term * (self.prior + 0.5)
    #
    #     return q_value + exploration_term, {'value': self.value,
    #                                         'visited': self.visited,
    #                                         'q_value': q_value,
    #                                         'exploration_term': exploration_term,
    #                                         'prior': self.prior}

    # def select_highest_ucb_child(self, c):
    #     max_child = max(self.children, key=lambda ch: ch.get_uct(c, self.visited)[0])
    #     mapped_children_DEBUG = list(map(lambda ch: ch.get_uct(c, self.visited)[1], self.children))
    #     return max_child
    #
    # def get_uct(self, c, parent_visits):
    #     if self.visited == 0:
    #         q_value = 0
    #     else:
    #         # q_value = self.value / self.visited
    #         avg_value = self.value / self.visited
    #         q_value = avg_value * 4
    #     exploration_term = c * (math.sqrt(parent_visits) / (self.visited + 1))
    #     # q_value *= (self.prior + 0.5)
    #     exploration_term *= (self.prior + 0.075)
    #     # explo_biased = exploration_term * (self.prior + 0.5)
    #
    #     return q_value + exploration_term, {'value': self.value,
    #                                         'visited': self.visited,
    #                                         'q_value': q_value,
    #                                         'exploration_term': exploration_term,
    #                                         'prior': self.prior}

    def select_highest_ucb_child(self, c):
        if self.visited > 2 * len(self.children):
            log_visited = math.log(self.visited)
            max_child = max(self.children, key=lambda ch: ch.get_uct_log(c, log_visited))
            # mapped_children_DEBUG = list(map(lambda ch: ch.get_uct_log_debug(c, log_visited)[1], self.children))
        else:
            max_child = max(self.children, key=lambda ch: ch.get_uct(c, self.visited))
            # mapped_children_DEBUG = list(map(lambda ch: ch.get_uct_debug(c, self.visited)[1], self.children))
        return max_child

    def get_uct_log(self, c, log_visited):
        if self.visited == 0:
            q_value = 0
        else:
            avg_value = self.value / self.visited
            q_value = avg_value * 4
        exploration_term = c * (math.sqrt(log_visited / (self.visited + 1)))

        exploration_term *= (self.prior + 0.075)
        return q_value + exploration_term

    def get_uct(self, c, parent_visits):
        if self.visited == 0:
            q_value = 0
        else:
            avg_value = self.value / self.visited
            q_value = avg_value
        exploration_term = c * (math.sqrt(parent_visits) / (self.visited + 1))

        exploration_term *= (self.prior + 0.075)
        return q_value + exploration_term

    def get_uct_log_debug(self, c, log_visited):
        if self.visited == 0:
            q_value = 0
        else:
            avg_value = self.value / self.visited
            q_value = avg_value * 4
        exploration_term = c * (math.sqrt(log_visited / (self.visited + 1)))

        exploration_term *= (self.prior + 0.075)
        return q_value + exploration_term, {'value': self.value,
                                            'visited': self.visited,
                                            'q_value': q_value,
                                            'exploration_term': exploration_term,
                                            'prior': self.prior}

    def get_uct_debug(self, c, parent_visits):
        if self.visited == 0:
            q_value = 0
        else:
            avg_value = self.value / self.visited
            q_value = avg_value
        exploration_term = c * (math.sqrt(parent_visits) / (self.visited + 1))

        exploration_term *= (self.prior + 0.075)
        return q_value + exploration_term, {'value': self.value,
                                            'visited': self.visited,
                                            'q_value': q_value,
                                            'exploration_term': exploration_term,
                                            'prior': self.prior}


class MCTS(ModelInterface):

    def __init__(self, name, model, max_iter=math.inf, max_time=math.inf,
                 uct_exploration_const=2.0, verbose=0, dirichlet_epsilon=0.2,
                 initial_alpha=0.4, final_alpha=0.1, decay_steps=50):
        super().__init__(name)
        self.root = None  # Node(game.get_snapshot())
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

    def set_root_new(self, game):
        """create new root Node."""
        game_copy = game.get_snapshot()
        encoded_state = game_copy.get_encoded_state()
        policy, _ = self.run_model(encoded_state, game_copy, add_noise=True)
        self.root = Node(game_copy, None, visited=1)
        self.root.explore_all_children(policy)

    def predict_best_move(self, game: Othello, deterministic=True):
        self.set_root_new(game)
        self.mcts_search()
        # print(f'\nafter simulating: {dict(self.root.get_all_next_move_counter())}')
        gc.collect()

        if deterministic:
            return self.best_moves(), None

        action_probs = np.zeros(ALL_FIELDS_SIZE)
        for child in self.root.children:
            move = child.move
            encoded_move = Othello.get_encoded_field(move)
            action_probs[encoded_move] = child.visited
        action_probs /= np.sum(action_probs)
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

        return best_items

    def best_moves(self):
        items = self.best_move_child_items()  # [(move, child),..]
        return [el[0] for el in items]  # [move1, move2, ...]

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
                encoded_state = node.game.get_encoded_state()
                # policy, value = self.model(
                #     torch.tensor(encoded_state, device=self.model.device).unsqueeze(0)
                # )
                # policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                # valid_moves = node.game.valid_moves_encoded()
                #
                # policy *= valid_moves
                # policy /= np.sum(policy)
                policy, value = self.run_model(encoded_state, node.game)
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

    @torch.no_grad()
    def run_model(self, encoded_state, game, add_noise=False):
        policy, value = self.model(
            torch.tensor(encoded_state, device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()

        if add_noise:
            policy = policy * (1 - self.dirichlet_epsilon) + self.dirichlet_epsilon \
                     * np.random.dirichlet([self.get_dirichlet_alpha()] * ALL_FIELDS_SIZE)

        valid_moves = game.valid_moves_encoded()
        policy *= valid_moves
        policy /= np.sum(policy)

        return policy, value

    def get_dirichlet_alpha(self):
        # Linear decay of alpha
        training_step = self.model.iterations_trained
        if training_step >= self.decay_steps:
            return self.final_alpha
        else:
            return self.initial_alpha - (self.initial_alpha - self.final_alpha) * (training_step / self.decay_steps)

    def mcts_iter(self):
        node, value = self.select_expansion_sim()
        self.backprop(node, value)

    def mcts_search(self):
        if self.max_time == math.inf and self.max_iter == math.inf:
            raise ValueError("At least one of max_time or max_iter must be specified.")

        start_time = time.perf_counter()

        iterations = 1  # root is always called in advance to add some noise
        while True:
            # Perform MCTS steps: selection, expansion, simulation, backpropagation
            self.mcts_iter()
            iterations += 1

            if iterations > 80 and self.root.game.turn > 51:
                pass

            if iterations > 83 and self.root.game.turn > 53:
                pass

            if iterations > 86 and self.root.game.turn > 55:
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
