import os
import math
import multiprocessing
import random
import numpy as np

# from reversi_game.agents.model_interface import ModelInterface
from .montecarlo import MCTS
from collections import Counter
from contextlib import contextmanager


def process_init(kwargs):
    global mcts_instance

    random_data = os.urandom(8)
    seed = int.from_bytes(random_data, byteorder="big")
    seed = seed % (2 ** 32 - 1)

    random.seed(seed)
    np.random.seed(seed)  # set so every process dont generate same simulations

    mcts_instance = MCTS(**kwargs)


def worker(game):
    _ = mcts_instance.simulate(game)
    move_counter = mcts_instance.root.get_all_next_move_counter()
    return move_counter, mcts_instance.iter_per_cycle()


class PMCTS:
    def __init__(self, time_limit=math.inf, iter_limit=math.inf, c=1.41, verbose=0):
        if time_limit is None and iter_limit is None:
            raise Exception(f'Need to give at least one limit in PMCTS!')
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.c = c
        self.pool = None
        self.num_processes = None
        self.verbose = verbose
        self.counter_best_moves = None

    def clean(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None  # Reset to None after cleanup
            self.num_processes = None

    def open_pool(self, num_processes=os.cpu_count()):
        self.pool = multiprocessing.Pool(num_processes,
                                         initializer=process_init,
                                         initargs=({'max_time': self.time_limit,
                                                    'max_iter': self.iter_limit,
                                                    'uct_exploration_const': self.c},))
        self.num_processes = num_processes

    # @staticmethod
    # @contextmanager
    # def create_pool_manager(pmcts, num_processes=os.cpu_count()):
    #     pmcts.pool = multiprocessing.Pool(num_processes,
    #                                       initializer=process_init,
    #                                       initargs=({'max_time': pmcts.time_limit,
    #                                                  'max_iter': pmcts.iter_limit,
    #                                                  'uct_exploration_const': pmcts.c},))
    #     pmcts.num_processes = num_processes
    #     try:
    #         yield  # self
    #     finally:
    #         if pmcts.pool:
    #             pmcts.pool.close()
    #             pmcts.pool.join()
    #             pmcts.pool = None  # Reset to None after cleanup
    #             pmcts.num_processes = None

    def best_moves(self):
        best_move = self.counter_best_moves.most_common(1)[0]  # ((7, 5), 3455)
        # print(f'best move: {best_move})')
        return [best_move[0]]

    def simulate(self, game):
        if self.pool is None:
            raise Exception("PMCTS pool is not initialized!.")

        counter_and_iteration_list = self.pool.starmap(worker, [(game,)
                                                                for _ in range(self.num_processes)])

        counter_list = []
        iter_list = []
        for counter, iteration_per_sec in counter_and_iteration_list:
            counter_list.append(counter)
            iter_list.append(iteration_per_sec)

        total_counters = sum(counter_list, Counter())
        # print(dict(total_counters))

        if self.verbose:
            print(f'game turn: {game.turn}')
            print(f'Iterations per sec: {sum(iter_list)}\n')

        # Find the key with the highest count
        self.counter_best_moves = total_counters
        # best_move = total_counters.most_common(1)[0]  # ((7, 5), 3455)
        # print(f'best move: {best_move})')
        # return [best_move[0]], None

        action_probs = np.zeros(game.action_space())
        for move, visited in total_counters.items():
            encoded_move = game.__class__.get_encoded_field(move)
            action_probs[encoded_move] = visited
        action_probs /= np.sum(action_probs)
        return action_probs

    def __repr__(self):
        return self.name + f' {self.iter_limit}'
