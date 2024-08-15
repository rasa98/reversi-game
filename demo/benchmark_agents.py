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
                             mcts_agent_500, load_parallel_mcts_agent_by_depth,
                             xyz_depth_1, xyz_depth_dyn)

# -------------------How to test two agents-----------------------------#
#bench_both_sides(minmax_human_depth_dyn, minmax_ga_depth_dyn, times=50, verbose=2)
#bench_both_sides(xyz_depth_dyn, minmax_ga_depth_dyn, times=20, verbose=2)

bench_both_sides(minmax_ga_depth_dyn, best_mlp_ppo, times=10, verbose=2)
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
