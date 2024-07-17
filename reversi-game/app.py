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

if __name__ == '__main__':
    log_filename = 'elo_after'
    log_folder_name = 'elo outputs/4'
    rounds = 100
    verbose = 0
    banned_agent_pairs = {(best_ars, cnn_trpo)}
    t = Tour(agents,
             log_filename,
             log_dir=os.path.join(os.getcwd(), log_folder_name),
             rounds=rounds,
             save_nth=10,
             verbose=verbose,
             banned=banned_agent_pairs)
    t.simulate()





