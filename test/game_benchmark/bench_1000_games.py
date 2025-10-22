import random
from reversi_game.game_modes import ai_vs_ai_cli
from test.game_benchmark.game_logic_v2 import Othello as Othello2
from reversi_game.agents.agent_interface import ai_random
import timeit

random.seed(0)
number = 1000
f = lambda: ai_vs_ai_cli(ai_random, ai_random, Othello2())
execution_time = timeit.timeit(f, number=number)
print(f"{number} games needs - {execution_time} sec")
