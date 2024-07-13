import os
import sys

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

relative_proj_dir_path = '../reversi-game/'
source_dir = os.path.abspath(os.path.join(os.getcwd(), relative_proj_dir_path))
sys.path.append(source_dir)
os.chdir(relative_proj_dir_path)



from gui.reversi_gui import OthelloGameGui

# Import any premade agent from 'read_all_agents' module
from read_all_agents import (alpha_200,
                             alpha_30,
                             best_ars,
                             best_mlp_ppo,
                             minmax_ga_best_depth_1,
                             mcts_agent_500)

game = OthelloGameGui(min_turn_time=2)

# if you wanna play against 'best_mlp' as a second turn player
game.play_human_vs_ai(best_mlp_ppo, 1)

# if you wanna visually watch two agents playing
# game.play_ai_vs_ai(alpha_200, mcts_agent_500)
