from models.minmax import Minimax
from heuristics.heu1 import heuristic, heuristic2
from game_modes import ai_vs_ai_cli
from collections import Counter
import timeit


def benchmark(ai1, ai2, times=50):
    vals = []
    for _ in range(times):
        winner_str = ai_vs_ai_cli(ai1, ai2)
        vals.append(winner_str)
    print(dict(Counter(vals)))


if __name__ == '__main__':
    mm1 = Minimax(3, heuristic)
    mm2 = Minimax(3, heuristic2)

    ai1 = {"name": "maximizer", "first_turn": True, "f": mm1.get_fields_and_estimate}
    ai2 = {"name": "minimizer", "first_turn": False, "f": mm2.get_fields_and_estimate}

    # ai_vs_ai_cli(ai1, ai2)

    # execution_time = timeit.timeit(lambda: ai_vs_ai_cli(ai1, ai2), number=200)  # Number of executions
    execution_time = timeit.timeit(lambda: benchmark(ai1, ai2, times=100), number=1)

    print(f"\nExecution time: {execution_time:.6f} seconds")
