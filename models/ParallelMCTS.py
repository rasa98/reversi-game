import os
import math
import multiprocessing
from multiprocessing import Manager

from .model_interface import ModelInterface
from .montecarlo import MCTS
from collections import Counter
from contextlib import contextmanager


def process_init(kwargs):
    global mcts_instance
    mcts_instance = MCTS(f'process local mcts', **kwargs)


def worker(game):
    _ = mcts_instance.predict_best_move(game)
    move_counter = mcts_instance.root.get_all_next_move_counter()
    return move_counter, mcts_instance.iter_per_cycle()


class PMCTS(ModelInterface):
    def __init__(self, name, time_limit=math.inf, iter_limit=math.inf):
        if time_limit is None and iter_limit is None:
            raise Exception(f'Need to give at least one limit in PMCTS!')
        super().__init__(name)
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.pool = None
        self.num_processes = None

    @staticmethod
    @contextmanager
    def create_pool_manager(pmcts, num_processes=os.cpu_count()):
        pmcts.pool = multiprocessing.Pool(num_processes,
                                          initializer=process_init,
                                          initargs=({'max_time': pmcts.time_limit,
                                                     'max_iter': pmcts.iter_limit},))
        pmcts.num_processes = num_processes
        try:
            yield  # self
        finally:
            if pmcts.pool:
                pmcts.pool.close()
                pmcts.pool.join()
                pmcts.pool = None  # Reset to None after cleanup
                pmcts.num_processes = None

    def predict_best_move(self, game):
        if self.pool is None:
            raise Exception("PMCTS object pool is equal to None.")

        counter_and_iteration_list = self.pool.starmap(worker, [(game,)
                                                                for _ in range(self.num_processes)])

        counter_list = []
        iter_list = []
        for counter, iteration_per_sec in counter_and_iteration_list:
            counter_list.append(counter)
            iter_list.append(iteration_per_sec)

        total_counters = sum(counter_list, Counter())
        # print(dict(total_counters))

        print(f'game turn: {game.turn}')
        print(f'Iterations per sec: {sum(iter_list)}\n')

        # Find the key with the highest count
        best_move = total_counters.most_common(1)[0]  # ((7, 5), 3455)
        # print(f'best move: {best_move})')
        return [best_move[0]], None
