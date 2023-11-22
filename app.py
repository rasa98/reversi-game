from models.minmax import Minimax
from heuristics.heu1 import heuristic, heuristic2
from game_modes import ai_vs_ai_cli
import timeit


def benchmark(ai1, ai2, times=50):
    pass


if __name__ == '__main__':
    mm1 = Minimax(4, heuristic)
    mm2 = Minimax(4, heuristic2)

    ai1 = {"name": "ai", "first_turn": True, "f": mm1.get_fields_and_estimate}
    ai2 = {"name": "bot_heu2", "first_turn": False, "f": mm2.get_fields_and_estimate}

    #ai_vs_ai_cli(ai1, ai2)

    execution_time = timeit.timeit(lambda: ai_vs_ai_cli(ai1, ai2), number=10)  # Number of executions

    print(f"\nExecution time: {execution_time:.6f} seconds")
