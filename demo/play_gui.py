import os
import sys

import warnings

warnings.filterwarnings('ignore', category=UserWarning)

relative_proj_dir_path = '../reversi-game/'
source_dir = os.path.abspath(os.path.join(os.getcwd(), relative_proj_dir_path))
sys.path.append(source_dir)
os.chdir(relative_proj_dir_path)

from gui.reversi_gui import (play_human_vs_ai,
                             play_ai_vs_ai)

# Import any premade agent from 'read_all_agents' module
from read_all_agents import (alpha_200,
                             alpha_30,
                             best_ars,
                             best_mlp_ppo,
                             cnn_trpo,
                             minmax_ga_best_depth_1,
                             mcts_agent_500,
                             mcts_agent_30,
                             ai_random,
                             load_mcts_agent_by_depth,
                             load_parallel_mcts_agent_by_depth
                             )

# if you wanna play against 'best_mlp' as a second turn player
# play_human_vs_ai(alpha_200, human_turn=2, min_turn_time=2, verbose=1)

# if you wanna visually watch two agents playing
play_ai_vs_ai(cnn_trpo, best_ars, min_turn_time=0, verbose=1)
