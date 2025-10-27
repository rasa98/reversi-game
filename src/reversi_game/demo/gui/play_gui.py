import warnings

warnings.filterwarnings('ignore', category=UserWarning)
#
# relative_proj_dir_path = '../reversi-game/'
# source_dir = os.path.abspath(os.path.join(os.getcwd(), relative_proj_dir_path))
# sys.path.append(source_dir)
# os.chdir(relative_proj_dir_path)

from reversi_game.visualization import (play_human_vs_ai)

# Import any premade agent from 'read_all_agents' module
from reversi_game.read_all_agents import (alpha_30,
                                          load_azero_agent_by_depth
                                          )
# Make alphazero agent with diff params like deeper search
alpha_1000 = load_azero_agent_by_depth(iter_depth=1000, c=1.73)


# if you wanna play against 'best_mlp' as a second turn player
play_human_vs_ai(alpha_30, human_turn=2, min_turn_time=2, verbose=2)

# if you wanna visually watch two agents playing
#play_ai_vs_ai(minmax_ga_best_depth_1, best_mlp_ppo , min_turn_time=0, verbose=1)

#play_human_vs_human(verbose=2)
