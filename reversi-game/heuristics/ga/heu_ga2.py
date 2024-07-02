import itertools
import random
from functools import partial

import numpy as np

from heuristics.ga.heu_func2 import (CountChips,
                                     CountDangerEarlyGame,
                                     CountCorners,
    # CountSaferEarlyGame,
                                     MaximizeMyMoves)

HEU_FUNCTION_CLASSES = (CountChips,
                        CountDangerEarlyGame,
                        CountCorners,
                        # CountSaferEarlyGame,
                        MaximizeMyMoves)


def create_heuristic(heu_func_param_dict):
    if len(heu_func_param_dict) == 0:
        raise Exception('heu_func_param_dict is empty...')

    return partial(heu, heu_func_param_dict)


def heu(heu_func_param_dict, game):
    res = 0
    for obj in heu_func_param_dict.values():
        res += obj.evaluate_state(game)
    return res


class HeuFuncIndividual:
    id_obj = itertools.count()

    def __init__(self, heuclass_to_instance, gen=1):  # , won=None, lost=None):
        """ params = chip_divisor, corner_divisor, corner_exponent, danger_divisor"""
        self.id = next(HeuFuncIndividual.id_obj)
        self.gen = gen
        self.heuclass_to_instance = heuclass_to_instance

    def copy(self):
        dict_copy = {k: v.copy() for k, v in self.heuclass_to_instance.items()}
        return HeuFuncIndividual(dict_copy,
                                 gen=self.gen)

    @staticmethod
    def create():
        res = {}
        heu_class_chosen = random.sample(HEU_FUNCTION_CLASSES,
                                         random.randint(1, len(HEU_FUNCTION_CLASSES)))
        for heu_class in heu_class_chosen:
            res[heu_class] = heu_class.create()  # heu class -> instace of it
        return HeuFuncIndividual(res)

    def mutate(self):  # TODO add missing heuFuct maybe, or remove some
        self = self.copy()
        # mutated = False
        for heu_obj in self.heuclass_to_instance.values():
            # if random.randint(0, 1):
            heu_obj.mutate()
            # mutated = True

        # if not mutated or random.randint(0, 1):
        if random.randint(0, 1):
            self.add_or_remove_heu_func()

        self.gen = 1
        return self

    def add_or_remove_heu_func(self):
        missing_heu_func = set(HEU_FUNCTION_CLASSES) - set(self.heuclass_to_instance.keys())
        max_num_of_missing = len(HEU_FUNCTION_CLASSES) - 1
        if len(missing_heu_func) == 0:
            self.remove_heu_random()
        elif len(missing_heu_func) == max_num_of_missing:
            self.add_heu_random()
        else:
            x = random.randint(0, 2)
            if x == 0:
                self.remove_heu_random()
            elif x == 1:
                self.add_heu_random()
            else:
                self.remove_heu_random()
                self.add_heu_random()  # can bring back removed, but it will have diff values

    def remove_heu_random(self):
        to_remove = list(self.heuclass_to_instance.keys())
        remove = random.choice(to_remove)
        del self.heuclass_to_instance[remove]

    def add_heu_random(self):
        to_add = [x for x in HEU_FUNCTION_CLASSES if x not in self.heuclass_to_instance.keys()]
        add = random.choice(to_add)
        self.heuclass_to_instance[add] = add.create()

    def crossover(self, other):
        this_copy = self.copy()
        a = this_copy.heuclass_to_instance
        b = other.heuclass_to_instance
        common_keys = set(a.keys()) & set(b.keys())
        if len(common_keys) == 0:
            deep_copy_b = {k: v.copy() for k, v in b.items()}
            all_heu_list = list(a.items()) + list(deep_copy_b.items())
            some_heu_list = random.sample(all_heu_list,
                                          random.randint(2, len(all_heu_list)))
            this_copy.heuclass_to_instance = dict(some_heu_list)
        else:
            if len(a) == 1 and len(b) == 1:
                this_copy.add_heu_random()

            max_crossover = len(common_keys)
            if len(common_keys) == len(a) and len(a) == len(b):  # so not for a to become basicly b
                max_crossover -= 1
            for heu_class in random.sample(list(common_keys), random.randint(1, max_crossover)):
                a[heu_class].crossover(b[heu_class])

        this_copy.gen = 1  # (self.gen + other.gen) // 2  # TODO reset to 1
        return this_copy

    @classmethod
    def selection(cls, list_of_elements, id_to_score_desc, survive_num=10, rates=(0.5, 0.8)):

        num_of_elements = len(list_of_elements)

        # survive_rate, crossover_rate = rates
        interesting_population_ratio, crossover_ratio = rates
        assert crossover_ratio * num_of_elements <= num_of_elements - survive_num, \
            "crossover ratio * num_of_elements >= num_of_elements - survive_num. Reduce it."

        res = []

        limit = interesting_population_ratio * num_of_elements
        id_to_el = {el.id: el for el in list_of_elements}

        interesting_population = []
        for idx, (k, v) in enumerate(id_to_score_desc.items()):
            elem = id_to_el[k]
            if idx < limit:  # TODO: on the edge el have same score
                if idx < survive_num:
                    elem.gen += 1
                    res.append(elem)
                interesting_population.append(elem)
            else:
                break

        #  crossover
        unique_pairs = set()
        num_of_crossover = int(crossover_ratio * num_of_elements)
        # half_num_of_crossover = (num_of_crossover // 2)
        for idx in range(num_of_crossover):
            # if idx < half_num_of_crossover:
            #     el = random.choice(survive)
            #     res.append(el.crossover(HeuFuncIndividual.create()))  # add some variability win new elements
            #     continue

            pair = frozenset(random.sample(interesting_population, 2))
            while pair in unique_pairs:
                pair = frozenset(random.sample(interesting_population, 2))
            el1, el2 = tuple(pair)
            unique_pairs.add(pair)
            res.append(el1.crossover(el2))

        #  mutate + some new population
        number_of_rest = num_of_elements - len(res)
        number_of_randoms = number_of_rest // 4
        for _ in range(number_of_randoms):
            res.append(HeuFuncIndividual.create())

        els = random.sample(interesting_population, number_of_rest - number_of_randoms)
        res += [el.mutate() for el in els]

        return res

    def get_heuristic(self):
        return create_heuristic(self.heuclass_to_instance)  # TODO: mutate i crossover imali su params

    def __str__(self):
        res = ""
        for k, v in self.heuclass_to_instance.items():
            res += str(v) + ', '

        return res[:-2]  # remove , from last


if __name__ == '__main__':
    pass
