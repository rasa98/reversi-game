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
                             mcts_agent_500)

bench_both_sides(minmax_ga_best_depth_1,
                 best_mlp_ppo,
                 times=10,
                 verbose=1)
