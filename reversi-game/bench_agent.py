import time
from game_modes import ai_vs_ai_cli
from collections import Counter
from tqdm import trange


def print_results(player1, player2, d):
    print(f'{"-" * 5} {player1} vs {player2} {"-" * 5}')
    print(d, "\n")


def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'Time needed: {end - start} seconds')
        print('----------------------------------------------', flush=True)
        return res

    return wrapper


def benchmark(ai1, ai2, times=200, verbose=1):
    my_range = range
    if verbose > 1:
        my_range = trange

    vals = []
    for _ in my_range(times):
        game = ai_vs_ai_cli(ai1, ai2)
        winner_str = game.get_winner()
        vals.append(winner_str)
    counter = Counter(vals)
    d = {}
    d[f"{ai1}"] = counter[1]
    d[f"{ai2}"] = counter[2]
    d[f"draw"] = counter[0]

    if verbose:
        print_results(ai1, ai2, d)
    return d


def benchmark_both_sides_last_board_state(ai1, ai2, times=200, verbose=1):
    """Check how many different finish states are. Good if all are diff.
       Some agents play strictly 'same' moves, so if pair two of that type,
       they can play 'same' games."""
    my_range = range
    if verbose > 1:
        my_range = trange

    vals = []
    for _ in my_range(times):
        game = ai_vs_ai_cli(ai1, ai2)
        board = tuple(game.board.reshape(-1))
        vals.append(board)
    counter = Counter(vals)
    frequency_list = list(counter.values())
    print(frequency_list)
    vals = []
    for _ in my_range(times):
        game = ai_vs_ai_cli(ai2, ai1)
        board = tuple(game.board.reshape(-1))
        vals.append(board)
    counter = Counter(vals)
    frequency_list = list(counter.values())
    print(frequency_list)


def _bench_both_sides(ai1, ai2, times=200, verbose=1):
    d1 = benchmark(ai1, ai2, times=times, verbose=verbose)
    d2 = benchmark(ai2, ai1, times=times, verbose=verbose)

    w_sum_1 = d1[ai1.name] + d2[ai1.name]
    w_sum_2 = d1[ai2.name] + d2[ai2.name]

    return w_sum_1, w_sum_2


def bench_both_sides(ai1, ai2, times=200, verbose=0):
    f = _bench_both_sides
    return f(ai1,
             ai2,
             times=times,
             verbose=verbose)
