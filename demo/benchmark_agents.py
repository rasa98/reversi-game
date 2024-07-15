import os
import sys

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

relative_proj_dir_path = '../reversi-game/'
source_dir = os.path.abspath(os.path.join(os.getcwd(), relative_proj_dir_path))
sys.path.append(source_dir)
os.chdir(relative_proj_dir_path)

from bench_agent import bench_both_sides

# Import any premade agent from 'read_all_agents' module
from read_all_agents import (alpha_200,
                             alpha_30,
                             best_ars,
                             mcts_agent_30,
                             best_mlp_ppo,
                             minmax_ga_best_depth_1,
                             mcts_agent_500, load_parallel_mcts_agent_by_depth)

bench_both_sides(alpha_200,
                 best_mlp_ppo,
                 times=100,
                 timed=True,
                 verbose=1)

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
