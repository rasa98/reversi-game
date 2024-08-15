import math
import numpy
import numpy as np
import time
import gc
# from game_logic import Othello
from game_modes import ai_vs_ai_cli
from agents.agent_interface import ai_random
from collections import Counter


# from .model_interface import ModelInterface


class Node:

    def __init__(self, game_copy, move, parent_node=None):
        self.game = game_copy
        self.visited = 0
        self.value = 0
        self.move = move
        self.children = []
        #self.move_to_child = {}  # TODO refactor to field children and every node has move that brought it to it.
        self.parent = parent_node
        self.valid_moves = list(game_copy.valid_moves_to_reverse)
        self.is_final_state = len(self.valid_moves) == 0

    def get_all_next_move_counter(self):
        """for debug"""
        #return Counter({move: child.visited for move, child in self.move_to_child.items()})
        return Counter({child.move: child.visited for child in self.children})

    def explored(self):
        return len(self.valid_moves) == 0

    def explore_new_child(self):
        move = self.valid_moves.pop()
        game_copy = self.game.get_snapshot()
        game_copy.play_move(move)

        child_node = Node(game_copy, move, self)
        self.children.append(child_node)
        #self.move_to_child[move] = child_node
        return child_node

    def select_highest_ucb_child(self, c):
        log_visited = math.log(self.visited)
        #max_child = max(self.move_to_child.values(), key=lambda ch: ch.get_uct(c, log_visited))
        max_child = max(self.children, key=lambda ch: ch.get_uct(c, log_visited))
        return max_child

    def get_uct(self, c, logged_parent_visits):
        avg_val = self.value / self.visited
        exploration_term = c * math.sqrt(logged_parent_visits / self.visited)
        return avg_val + exploration_term

    # def select_highest_ucb_child(self, c):
    #     max_child = max(self.move_to_child.values(), key=lambda ch: ch.get_uct(c, self.visited))
    #     return max_child
    #
    # def get_uct(self, c, par):
    #     avg_val = ((self.value / self.visited) + 1) / 2
    #     exploration_term = c * (math.sqrt(par) / self.visited)
    #     return avg_val + exploration_term

    def simulate_game(self):
        game_copy = self.game.get_snapshot()
        game = ai_vs_ai_cli(ai_random, ai_random, game_copy)  # 1, 2, or 0
        return game.get_winner()


class MCTS:

    def __init__(self, max_iter=math.inf, max_time=math.inf, uct_exploration_const=1.42, verbose=0):
        # super().__init__(name)
        self.root = None  # Node(game.get_snapshot())
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
        self.root = Node(game.get_snapshot(), None)

    def predict_best_move_OLD(self, game):  # TODO remove
        self.set_root_new(game)
        self.mcts_search()
        # print(f'\nafter simulating: {dict(self.root.get_all_next_move_counter())}')
        gc.collect()
        return self.best_moves(), None

    def simulate(self, game):
        self.set_root_new(game)
        self.mcts_search()
        # print(f'\nafter simulating: {dict(self.root.get_all_next_move_counter())}')
        gc.collect()

        action_probs = np.zeros(game.action_space())
        for child in self.root.children:
            encoded_move = game.__class__.get_encoded_field(child.move)
            action_probs[encoded_move] = child.visited
        action_probs /= np.sum(action_probs)
        return action_probs

        # return self.best_moves(), None

    # def best_move_child_item(self):
    #     return max(self.root.move_to_child.items(), key=lambda item: item[1].visited)

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
        while True:
            if node.is_final_state:
                break  # return node
            elif not node.explored():
                node = node.explore_new_child()  # return node.explore_new_child()
                break
            else:
                node = node.select_highest_ucb_child(self.uct_exploration_const)
        return node, node.simulate_game()  # returns (node , winner)

    def backprop(self, node: Node, winner: int):
        while node is not None:
            node.visited += 1
            if winner == node.game.last_turn:  # player that turned state into current nodes
                node.value += 1
            elif winner == 0:
                pass
            else:
                node.value -= 1
            node = node.parent

    @staticmethod
    def is_time_limit_reached(start_time, max_time_sec):
        elapsed_time = time.time() - start_time
        return elapsed_time >= max_time_sec

    def mcts_iter(self):
        node, winner = self.select_expansion_sim()
        self.backprop(node, winner)

    def mcts_search(self):
        if self.max_time == math.inf and self.max_iter == math.inf:
            raise ValueError("At least one of max_time or max_iter must be specified.")

        start_time = time.process_time()
        iterations = 0
        while True:
            # Perform MCTS steps: selection, expansion, simulation, backpropagation
            self.mcts_iter()
            iterations += 1

            # Check termination conditions
            elapsed_time = time.process_time() - start_time
            if elapsed_time >= self.max_time or iterations >= self.max_iter:
                break
        self.last_cycle_time = elapsed_time
        self.last_cycle_iteration = iterations

        if self.verbose:
            print(f'game turn: {self.root.game.turn}')
            print(f'iterations {iterations} per one cycle: {self.iter_per_cycle()}\n')

# time_limit = 1
# iter_limit = 30  # math.inf
# verbose = 0  # 0 means no logging
# mcts_model = MCTS(f'mcts iter_limit {iter_limit}',
#                   max_time=time_limit,
#                   max_iter=iter_limit,
#                   verbose=verbose)
