import math
import os
import sys

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

relative_proj_dir_path = '../reversi-game/'
source_dir = os.path.abspath(os.path.join(os.getcwd(), relative_proj_dir_path))
sys.path.append(source_dir)
os.chdir(relative_proj_dir_path)

from bench_agent import (benchmark,
                         bench_both_sides,
                         benchmark_both_sides_last_board_state)

# Import any premade agent from 'read_all_agents' module
from read_all_agents import (alpha_200,
                             alpha_30,
                             best_ars,
                             cnn_trpo,
                             ppo_cnn,
                             mcts_agent_30,
                             best_mlp_ppo,
                             ai_random,
                             minmax_ga_best_depth_1,
                             minmax_ga_depth_dyn,
                             minmax_human_depth_1,
                             minmax_human_depth_dyn,
                             mcts_agent_500, load_parallel_mcts_agent_by_depth, load_mcts_model
                             )

pmcts_30 = load_parallel_mcts_agent_by_depth(30)


# -------------------------------------------------------------------------
def f(current_turn):
    if current_turn < 30:
        # Before turn 30, keep iterations low, e.g., 200
        return int(1000 + ((current_turn - 0) / 29) * (2000 - 1000))
    elif 30 <= current_turn <= 37:
        # Linearly increase from 200 to 800 iterations between turns 30 and 37
        return int(2000 + ((current_turn - 30) / 7) * (3000 - 2000))
    elif 38 <= current_turn <= 60:
        # Linearly increase from 800 to 10000 iterations between turns 38 and 60
        return int(3000 + ((current_turn - 37) / 23) * (20000 - 3000))
    else:
        # After turn 60, keep iterations at 10000
        return 10000


mcts_params = {'max_time': math.inf,
               'max_iter': math.inf,
               'f_iter_per_turn': f,
               'c': 1.41,
               'verbose': 0}
mcts = load_mcts_model(params=mcts_params)

# -------------------How to test two agents-----------------------------#

# bench_both_sides(minmax_human_depth_dyn, minmax_ga_depth_dyn, times=50, verbose=2)
# bench_both_sides(xyz_depth_dyn, minmax_ga_depth_dyn, times=20, verbose=2)

bench_both_sides(mcts, alpha_30, times=10, verbose=2)

# bench_both_sides(minmax_human_depth_1, best_ars, times=100, verbose=2)
# bench_both_sides(minmax_human_depth_1, ppo_cnn, times=100, verbose=2)

# ------------------Showing how to run parallel mcts----------------------#

# pmcts_agent_30 = load_parallel_mcts_agent_by_depth(30)
# pmcts_agent_30.set_deterministic(False)
#
# try:
#     pmcts_agent_30.open_pool(num_process=4)  # need to manually open and close pool for parallel mcts#
#     from bench_agent import bench_both_sides
#
#     bench_both_sides(mcts_agent_30,
#                      pmcts_agent_30,
#                      times=100,
#                      timed=True,
#                      verbose=2)
#     # parallel_mcts_30 is 70:30 compared to mcts_30
# finally:
#     pmcts_agent_30.clean_pool()


# --------------------Testing for same move playing----------------------------#

# cnn_trpo and best_ars can play 'same moves' esspecially against each other
# ppo_mlp does it much more rarely like less than 5 %

# benchmark_both_sides_last_board_state(cnn_trpo,
#                                       best_ars,
#                                       times=100,
#                                       verbose=2)
