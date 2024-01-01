import numpy as np
import random, itertools
from heuristics.heu1 import (count_chips, count_corners, count_danger_early_game,
                             minimize_opponent_moves)


def create_heuristic(chip_divisor, corner_divisor, corner_exponent, danger_divisor, min_opp_score):
    """
        chip_divisor      - [1, 2, 3..., 20]
        corner_divisor    - [1, 2, 3..., 20]
        corner_exponent   - range [1.0 - 3.0]
        danger_divisor    - [1, 2, 3..., 20]
        min_opp_score     - 100 - 1000
    """

    def heu(game):
        stats = (game.chips, game.turn, game.board)
        res = (count_chips(stats, lambda turn: (turn // chip_divisor) + 1) +
               count_corners(stats, lambda x: ((60 - x) // corner_divisor) ** corner_exponent) +
               count_danger_early_game(stats, lambda turn: (65 - turn) // danger_divisor) +

               minimize_opponent_moves(game, min_opp_score)
               )
        return res

    return heu


class HeuristicChromosome:
    id_obj = itertools.count()

    def __init__(self, params, gen=1):  # , won=None, lost=None):
        """ params = chip_divisor, corner_divisor, corner_exponent, danger_divisor"""
        self.id = next(HeuristicChromosome.id_obj)
        self.params = params

        self.gen = gen

        # self.score = score
        # self.won = []
        # self.lost = []
        # if won is not None:
        #     self.won = won
        # if lost is not None:
        #     self.lost = lost

    def copy(self):
        return HeuristicChromosome(self.params,
                                   gen=self.gen
                                   # ,
                                   # won=self.won,
                                   # lost=self.lost
                                   )

    @staticmethod
    def create():
        params = (random.uniform(1.0, 20.0),
                  random.uniform(1.0, 20.0),
                  random.uniform(1.0, 3.0),
                  random.uniform(1.0, 20.0),
                  random.uniform(100.0, 1000.0))
        return HeuristicChromosome(params)

    def _param_delta(self):
        return random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), \
            random.uniform(-0.3, 0.3), random.uniform(-3.0, 3.0), \
            random.uniform(-100.0, 100.0)

    def bound_mutated(self, idx, val):
        match idx:
            case 2:
                return (val - 1) % 2 + 1
            case 4:
                return (val - 100) % 900 + 100
            case 1 | 3 | 0:
                return (val - 1) % 19 + 1  # [1 - 20] -> [0 - 19] -> % -> [1 - 20]
            case _:
                raise Exception('Param index mutation function not configured correctly or for that idx!!!')

    # def mutate(self):
    #     new_params = [x + y for x, y in zip(self._param_delta(), self.params)]
    #
    #     random_chromosome_bit_list = [random.randint(0, 1) for _ in range(len(self.params))]
    #     for idx, flag in enumerate(random_chromosome_bit_list):  # flag is 0 or 1 not a true/false
    #         if flag:
    #             new_params[idx] = self.mutate_bound(idx)
    #             # if isinstance(new_params[idx], complex):
    #             #     print(f'index - {idx}, num :{new_params[idx]}')
    #             #     raise Excption('Complex from mutate_f')
    #         else:
    #             new_params[idx] = self.params[idx]  # TODO to do this better
    #
    #     self.params = tuple(new_params)
    #     self.gen = 1
    #     return self

    # def mutate(self):
    #     delta = self._param_delta()
    #     new_params = list(self.params)
    #     random_chromosome_bit_list = [random.randint(0, 1) for _ in range(len(self.params))]
    #     for idx, flag in enumerate(random_chromosome_bit_list):  # flag is 0 or 1 not a true/false
    #         if flag:
    #             new_val = new_params[idx] + delta[idx]
    #             new_params[idx] = self.bound_mutated(idx, new_val)
    #
    #     self.params = tuple(new_params)
    #     self.gen = 1
    #     return self
    #

    def mutate(self):
        delta = self._param_delta()
        new_params = list(self.params)
        for idx, val in enumerate(new_params):  # flag is 0 or 1 not a true/false
            if random.randint(0, 1):
                new_val = val + delta[idx]
                new_params[idx] = self.bound_mutated(idx, new_val)

        self.params = tuple(new_params)
        self.gen = 1
        return self

    def crossover(self, noble):
        this_copy = self.copy()
        param_index_other = random.sample([0, 1, 2, 3, 4], random.randint(1, 4))
        res = list(this_copy.params)
        for i in param_index_other:
            res[i] = noble.params[i]
        this_copy.params = tuple(res)
        # this_copy = this_copy.mutate()  # can i mutate this then???
        this_copy.gen = (self.gen + noble.gen) // 2
        return this_copy

    @classmethod
    def selection(cls, list_of_elements, id_to_score_desc, rates=(0.5, 0.25)):
        if rates[0] + rates[1] > 0.999:
            raise Exception('Selection rates are not configured correctly. Sum must be less than 1')

        survive_rate, crossover_rate = rates
        num_of_elements = len(list_of_elements)
        res = []

        id_to_el = {el.id: el for el in list_of_elements}
        survive, mutate = [], []
        for idx, (k, v) in enumerate(id_to_score_desc.items()):
            elem = id_to_el[k]
            if idx < survive_rate * num_of_elements:  # TODO: on the edge el have same score
                elem.gen += 1
                survive.append(elem)
            else:
                mutate.append(elem)

        res += survive

        #  crossover
        unique_pairs = set()
        num_of_crossover = int(crossover_rate * num_of_elements)
        for idx in range(num_of_crossover):
            pair = tuple(random.sample(survive, 2))
            while pair in unique_pairs or (pair[1], pair[0]) in unique_pairs:
                pair = tuple(random.sample(survive, 2))
            el1, el2 = pair
            unique_pairs.add(pair)

            if idx < (num_of_crossover // 2):
                res.append(el1.crossover(HeuristicChromosome.create()))  # add some variability win new elements
            else:
                res.append(el1.crossover(el2))

        #  mutate + some new population
        number_of_rest = num_of_elements - len(res)
        number_of_randoms = number_of_rest // 2
        for _ in range(number_of_randoms):
            res.append(HeuristicChromosome.create())

        els = random.sample(mutate, number_of_rest - number_of_randoms)
        res += [el.mutate() for el in els]

        return res

    def get_heuristic(self):
        return create_heuristic(*self.params)

    def __repr__(self):
        return self.__dict__


if __name__ == '__main__':
    pass
