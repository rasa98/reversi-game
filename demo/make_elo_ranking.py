import os
import sys

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

cwd = os.getcwd()
relative_proj_dir_path = '../reversi-game/'
source_dir = os.path.abspath(os.path.join(cwd, relative_proj_dir_path))
sys.path.append(source_dir)
os.chdir(relative_proj_dir_path)

from elo_rating import Tournament as Tour

# Import any premade agent from 'read_all_agents' module
from read_all_agents import (agents,
                             best_ars,
                             best_mlp_ppo,
                             minmax_ga_best_depth_1,
                             cnn_trpo)

log_filename = 'elo'
log_folder_name = 'elo logs'  # outputs will be in this folder
rounds = 10
verbose = 0

# shouldnt play against each other, cuz they play 'same' matches
banned_agent_pairs = {(best_ars, cnn_trpo)}

t = Tour([best_ars, cnn_trpo, best_mlp_ppo, minmax_ga_best_depth_1],  # agents,
         log_filename,
         log_dir=os.path.join(cwd, log_folder_name),
         rounds=rounds,
         save_nth=5,
         verbose=verbose,
         banned=banned_agent_pairs)
t.simulate()
