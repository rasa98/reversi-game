import os
import sys

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# cwd = os.getcwd()
# relative_proj_dir_path = '../reversi-game/'
# source_dir = os.path.abspath(os.path.join(cwd, relative_proj_dir_path))
# sys.path.append(source_dir)
# os.chdir(relative_proj_dir_path)

from reversi_game.elo_rating import Tournament as Tour

# Import any premade agent from 'read_all_agents' module
from reversi_game.read_all_agents import (agents, minmmax_agents,
                             best_ars,
                             best_mlp_ppo,
                             minmax_ga_best_depth_1,
                             minmax_human_depth_1,
                             cnn_trpo,
                             mcts_agent_30,
                             pmcts_agent_30,
                             ai_random)

log_filename = 'elo minmax'
log_folder_name = 'elo logs'  # outputs will be in this folder
rounds = 100
verbose = 0

# shouldnt play against each other, cuz they play 'same' matches
banned_agent_pairs = {(best_ars, cnn_trpo)}

t = Tour(minmmax_agents,  # agents,
         log_filename,
         log_dir=os.path.join(cwd, log_folder_name),
         rounds=rounds,
         save_nth=5,
         verbose=verbose,
         banned=banned_agent_pairs)
#pmcts_agent_30.open_pool()
t.simulate()
#pmcts_agent_30.clean_pool()