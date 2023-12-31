import sys

sys.path.append('/home/rasa/PycharmProjects/reversiProject/')  # TODO fix this hack
from heuristics.heu2 import HeuristicChromosome
from models.minmax import Minimax
from game_modes import ai_vs_ai_cli
import random, itertools
import timeit, time
import concurrent.futures
from collections import defaultdict

random_seed = time.time()
population_size = 50
TOURNAMENTS = 100
ROUNDS = 20
CORES = 4
SAVE_FREQ = 10


def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


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


def generate_match_pairs(list_of_players, rounds):
    res = []
    for pair in generate_all_pairs(list_of_players, rounds):
        ai1, ai2 = pair
        res.append(pair)
        res.append((ai2, ai1))
    return res


def simulate_tournament(res):
    score = defaultdict(int)
    for pair in res:
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


def parallel_process_list(players, func, num_processes=1):
    match_pairs = generate_match_pairs(players, ROUNDS)
    # print(f'len of matches pairs: {len(match_pairs)}\n\n')

    chunk_size = len(match_pairs) // num_processes
    partitions = [match_pairs[i:i + chunk_size] for i in range(0, len(match_pairs), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Map the function to th

        results = list(executor.map(func, partitions))

    # print(f'cores: {num_processes} \nlista vracena: {results}\n')
    merged_dict = {p.id: 0 for p in players}

    for result_dict in results:
        for key, value in result_dict.items():
            merged_dict[key] += value

    sorted_dict = dict(sorted(merged_dict.items(), key=lambda item: item[1], reverse=True))

    # print(sorted_dict)
    return sorted_dict


def save_current_list(player_list, id_to_score, times):

    sorted_list = sorted(player_list, key=lambda obj: (id_to_score[obj.id], obj.gen), reverse=True)

    with open(f'dump_ga_models_bigger2/current_list_{times}.txt', 'w') as f:
        for el in sorted_list:
            f.write(f'gen alive: {el.gen}, params: {el.params}, id: {el.id}, score: {id_to_score[el.id]}\n')


def save_start_list(player_list):
    sorted_list = sorted(player_list, key=lambda obj: obj.params, reverse=True)

    with open(f'dump_ga_models_bigger2/start_list.txt', 'w') as f:
        for el in sorted_list:
            f.write(f'gen alive: {el.gen}, params: {el.params}, id: {el.id}\n')


random.seed(random_seed)
players = [HeuristicChromosome.create() for _ in range(population_size)]
inner_players = players
save_counter = 0
save_start_list(players)
for tour_num in range(1, TOURNAMENTS+1):
    id_to_score_desc = parallel_process_list(inner_players, simulate_tournament, num_processes=CORES)

    # clone = [p.id for p in inner_players]
    # print(f'No duplicates in list -> {len(set(clone)) == len(clone)}')

    # print(sorted(id_to_score_desc.values(), key=lambda x: x, reverse=True))
    # print()
    if tour_num % SAVE_FREQ == 0:
        save_counter += 1
        save_current_list(inner_players, id_to_score_desc, save_counter)
    inner_players = HeuristicChromosome.selection(inner_players, id_to_score_desc, rates=(0.4, 0.2))
    if tour_num % SAVE_FREQ == 0:
        print(f'Done {int(tour_num / TOURNAMENTS * 100)}%')




# executed_time = timeit.timeit(lambda: parallel_process_list(players,
#                                                             simulate_tournament,
#                                                             num_processes=CORES),
#                               number=1)
#
# print(f'time needed - {executed_time}\n')


# for cores in [1, 2, 4]:
#     executed_time = timeit.timeit(lambda: parallel_process_list(players,
#                                                                 simulate_tournament,
#                                                                 num_processes=cores),
#                                   number=1)
#     print(f'time needed for {cores} cores - {executed_time}\n')

# match_pairs = generate_match_pairs(players, ROUNDS)
# score = simulate_tournament(match_pairs)
# print(score)
