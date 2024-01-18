import sys
# sys.path.append('/home/rasa/PycharmProjects/reversiProject/')  # TODO fix this hack
import os

source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(source_dir)

from heuristics.ga.heu_ga import HeuFuncIndividual
from models.minmax import Minimax
from game_modes import ai_vs_ai_cli
import random, itertools
import timeit, time
from collections import defaultdict
from dask.bag import from_sequence
from dask import delayed, compute
from dask.distributed import Client

random_seed = time.time()
population_size = 50
TOURNAMENTS = 10
ROUNDS = 10
CORES = int(sys.argv[1]) #if len(sys.argv) >= 2 else os.cpu_count()
SAVE_FREQ = 2
SEL_CROSSOVER = (0.3, 0.5)
REMATCH = False
LOG_DIR = sys.argv[2]

os.makedirs(LOG_DIR, exist_ok=True)


def generate_all_pairs(ps, rounds):
    ps = list(ps)
    res = []
    l = len(ps)
    if l % 2 != 0 or rounds >= l:
        raise Exception("cant be odd number of population, or rounds >= than len of pop")

    res += [(ps[i], ps[l - 1 - i]) for i in range(l // 2)]
    for _ in range(rounds - 1):
        ps = [ps[0]] + [ps[-1]] + ps[1: -1]
        res += [(ps[i], ps[l - 1 - i]) for i in range(l // 2)]
    return res


def add_rematches(gen_matches):
    res = []
    for pair in gen_matches:
        ai1, ai2 = pair
        res.append(pair)
        res.append((ai2, ai1))
    return res


def simulate_tournament(matches):
    score = defaultdict(int)
    for pair in matches:
        x, y = pair
        mm1 = Minimax(lambda _: 1, x.get_heuristic())
        mm2 = Minimax(lambda _: 1, y.get_heuristic())
        ai1 = {"name": "bot1", "f": mm1.predict_best_move}
        ai2 = {"name": "bot2", "f": mm2.predict_best_move}

        winner = ai_vs_ai_cli(ai1, ai2)
        if winner != 0:
            winner_idx = winner - 1
            winner_model = pair[winner_idx]

            score[winner_model.id] += 3
        else:  # if its draw
            score[pair[0].id] += 1
            score[pair[1].id] += 1
    # print(dict(score))
    return score


def parallel_process_list(players, func, num_processes=1, rematch=True):
    match_pairs = generate_all_pairs(players, ROUNDS)
    if rematch:
        match_pairs = add_rematches(match_pairs)
    # print(f'len of matches pairs: {len(match_pairs)}\n\n')

    # # Manually partition the list of matches
    chunk_size = len(match_pairs) // num_processes
    partitions = [match_pairs[i:i + chunk_size] for i in range(0, len(match_pairs), chunk_size)]

    # Use Dask to parallelize the processing of partitions
    delayed_results = [delayed(func)(partition) for partition in partitions]

    # Combine the results
    merged_dict = {p.id: 0 for p in players}

    results = compute(*delayed_results,
                      #num_workers=num_processes,
                      #scheduler='distributed'
                      )

    for result_dict in results:
        for key, value in result_dict.items():
            merged_dict[key] += value

    sorted_dict = dict(sorted(merged_dict.items(), key=lambda item: item[1], reverse=True))

    # print(sorted_dict)
    return sorted_dict


def save_current_list(player_list, id_to_score, rounds):
    sorted_list = sorted(player_list, key=lambda obj: (id_to_score[obj.id], obj.gen), reverse=True)

    with open(f'{LOG_DIR}/board_after_{rounds}_tour.txt', 'w') as f:
        for el in sorted_list:
            f.write(f'gen alive: {el.gen}, params: {str(el)}, id: {el.id}, score: {id_to_score[el.id]}\n')


def save_start_list(player_list):
    sorted_list = player_list  # sorted(player_list, key=lambda obj: obj.params, reverse=True)

    with open(f'{LOG_DIR}/start_list.txt', 'w') as f:
        for el in sorted_list:
            f.write(f'gen alive: {el.gen}, params: {str(el)}, id: {el.id}\n')


def run_ga():
    random.seed(random_seed)
    players = [HeuFuncIndividual.create() for _ in range(population_size)]
    inner_players = players
    save_counter = 0
    save_start_list(players)

    start = time.perf_counter()

    for tour_num in range(1, TOURNAMENTS + 1):
        id_to_score_desc = parallel_process_list(inner_players, simulate_tournament,
                                                 num_processes=CORES, rematch=REMATCH)

        if tour_num % SAVE_FREQ == 0:
            save_counter += 1
            save_current_list(inner_players, id_to_score_desc, save_counter * SAVE_FREQ)

        inner_players = HeuFuncIndividual.selection(inner_players, id_to_score_desc, rates=SEL_CROSSOVER)

        if tour_num % SAVE_FREQ == 0:
            end = time.perf_counter()
            print(f'Done {int(tour_num / TOURNAMENTS * 100)}%. Time needed: {end - start:.2f}')
            start = end
            sys.stdout.flush()


if __name__ == '__main__':
    from dask_jobqueue import SLURMCluster
    cluster = SLURMCluster(        
        cores=12,
        processes=1,
        memory='8GB'
    )    

    print(f'starting script: cores - {CORES}')
    start = time.perf_counter()

    with Client(cluster) as client:        
        run_ga()

    end = time.perf_counter()
    print(f'Done in {end - start} seconds,'
          f' {(end - start) // 60} mins or'
          f' {(end - start) // 3600} hours')
