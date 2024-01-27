import os
import math
import multiprocessing
from .model_interface import ModelInterface
from .montecarlo import MCTS
from collections import Counter
from contextlib import contextmanager


def worker(game, mcts_list, mcts_id):
    mcts_instance = mcts_list[mcts_id]
    _ = mcts_instance.predict_best_move(game)
    move_counter = mcts_instance.root.get_all_next_move_counter()
    return move_counter


class PMCTS(ModelInterface):
    def __init__(self, name, time_limit=math.inf, iter_limit=math.inf):
        if time_limit is None and iter_limit is None:
            raise Exception(f'Need to give at least one limit in PMCTS!')
        super().__init__(name)
        self.time_limit = time_limit
        self.iter_limit = iter_limit
        self.pool = None
        self.mcts_list = None
        self.num_processes = None

    @contextmanager
    def create_pool_manager(self, num_processes=os.cpu_count()):
        self.pool = multiprocessing.Pool(num_processes)
        self.mcts_list = [MCTS(f'mcts {self.time_limit}s',
                               max_time=self.time_limit,
                               max_iter=self.iter_limit)
                          for _ in range(num_processes)]
        self.num_processes = num_processes
        try:
            yield self
        finally:
            if self.pool:
                self.pool.close()
                self.pool.join()
                self.pool = None  # Reset to None after cleanup
                self.mcts_list = None
                self.num_processes = None

    def predict_best_move(self, game):
        if self.pool is None or not self.mcts_list:
            raise Exception("PMCTS object pool or mcts_list is equal to None.")

        counter_list = self.pool.starmap(worker, [(game, self.mcts_list, i)
                                                  for i in range(self.num_processes)])
        total_counters = sum(counter_list, Counter())

        print(dict(total_counters))

        # Find the key with the highest count
        best_move = total_counters.most_common(1)[0]  # ((7, 5), 3455)
        # print(f'best move: {best_move})')
        return [best_move[0]], None
