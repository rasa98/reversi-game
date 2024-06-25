import itertools
import os
import random
import sys
# import time
import concurrent.futures
from collections import defaultdict
from tqdm import trange

if os.environ['USER'] != 'student':
    source_dir = os.path.abspath(os.path.join(os.getcwd(), '../../../'))
    sys.path.append(source_dir)

from heuristics.ga.heu_ga import HeuFuncIndividual
from models.MiniMaxAgent import load_minimax_agent
from game_modes import ai_vs_ai_cli


def generate_all_pairs(ps, rounds):
    ps = list(ps)
    res = []
    l = len(ps)

    assert l % 2 == 0, "cant be odd number of population"
    assert rounds < l, "rounds cant be >= than len of population"

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
    seed = get_seed()
    random.seed(seed)

    score = defaultdict(int)
    for pair in matches:
        x, y = pair
        mm1 = load_minimax_agent('bot 1', lambda _: 1, x.get_heuristic())
        mm2 = load_minimax_agent('bot 2', lambda _: 1, y.get_heuristic())

        winner = ai_vs_ai_cli(mm1, mm2)
        if winner != 0:
            winner_idx = winner - 1
            winner_model = pair[winner_idx]

            score[winner_model.id] += 3
        else:  # if its draw
            score[pair[0].id] += 1
            score[pair[1].id] += 1
    # print(dict(score))
    return score


def parallel_process_list(players, func, rematch=True):
    match_pairs = generate_all_pairs(players, ROUNDS)
    if rematch:
        match_pairs = add_rematches(match_pairs)
    # print(f'len of matches pairs: {len(match_pairs)}\n\n')

    chunk_size = len(match_pairs) // CORES
    partitions = [match_pairs[i:i + chunk_size] for i in range(0, len(match_pairs), chunk_size)]


    results = list(executor.map(func, partitions))

    # print(f'cores: {CORES} \nlista vracena: {results}\n')
    merged_dict = {p.id: 0 for p in players}

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


def get_seed():
    random_data = os.urandom(8)
    seed = int.from_bytes(random_data, byteorder="big")
    return seed % (2 ** 32 - 1)


def run_ga():
    players = [HeuFuncIndividual.create() for _ in range(population_size)]
    inner_players = players
    save_counter = 0
    save_start_list(players)
    for tour_num in trange(1, TOURNAMENTS + 1):
        id_to_score_desc = parallel_process_list(inner_players,
                                                 simulate_tournament,
                                                 rematch=REMATCH)

        if tour_num % SAVE_FREQ == 0:
            save_counter += 1
            save_current_list(inner_players, id_to_score_desc, save_counter * SAVE_FREQ)

        inner_players = HeuFuncIndividual.selection(inner_players, id_to_score_desc, rates=SEL_CROSSOVER)

        # if tour_num % SAVE_FREQ == 0:
        #     print(f'Done {int(tour_num / TOURNAMENTS * 100)}%')


if __name__ == "__main__":
    if os.environ['USER'] != 'student':
        os.chdir('../../../')
        CORES = 2
    else:
        CORES = int(os.environ['SLURM_CPUS_ON_NODE']) // 2

    population_size = 20#200
    TOURNAMENTS = 100#1000
    ROUNDS = 10#100

    SAVE_FREQ = 10
    SEL_CROSSOVER = (0.5, 0.4)
    REMATCH = False
    LOG_DIR = 'models_output/ga/train_logs/'

    os.makedirs(LOG_DIR, exist_ok=True)

    print(
        f"params:\n\npop: {population_size}\n"
        f"tournaments: {TOURNAMENTS}\nrounds: {ROUNDS}\n"
        f"ratio: {SEL_CROSSOVER}\nrematch: {REMATCH}\n"
        f"cores: {CORES}\n\n")

    # start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=CORES) as executor:
        run_ga()
    # end = time.perf_counter()
    # print(f'Done in {end - start} seconds,'
    #       f' {(end - start) // 60} mins or'
    #       f' {(end - start) // 3600} hours')
