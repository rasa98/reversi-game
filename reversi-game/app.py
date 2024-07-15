import math
import os
import random
import time

import numpy as np

from elo_rating import Tournament as Tour
from read_all_agents import (agents,
                             minmax_ga_best_depth_1,
                             minmax_human_depth_dyn,
                             best_ars,
                             cnn_trpo
                             )


def run_elo_ranking_tournament(agents):
    log_filename = 'elo_after'
    log_folder_name = 'elo outputs'
    rounds = 100
    verbose = 0
    banned_agent_pairs = {(best_ars, cnn_trpo)}
    t = Tour(agents,
             log_filename,
             log_dir=os.path.join(os.getcwd(), log_folder_name),
             rounds=rounds,
             save_nth=5,
             verbose=verbose,
             banned=banned_agent_pairs)
    t.simulate()


if __name__ == '__main__':
    # Use different seedsreversi-game/scripts/rl/train_model_ars.py
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    # for agent in agents:
    #     agent.set_deterministic(False)

    run_elo_ranking_tournament(agents)


    # from bench_agent import bench_both_sides
    # bench_both_sides(minmax_ga_best_depth_1, minmax_human_depth_dyn, times=10, verbose=1)


