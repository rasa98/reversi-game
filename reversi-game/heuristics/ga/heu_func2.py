import random
from abc import ABC, abstractmethod

from heuristics.heu1 import (count_chips,
                             count_corners,
                             count_danger_early_game,
                             count_safer,
                             max_my_moves, weighted_piece_counter)


class HeuFunctionInterface(ABC):
    def __init__(self, *params):
        self.params = list(params)

    def get_params(self):
        """return parameters"""
        return tuple(self.params)

    def copy(self):
        return self.__class__(*self.params)

    def mutate(self):
        """give some delta change to param/s"""
        indexes = range(len(self.params))
        for idx in indexes:
            val = self.delta(idx) + self.params[idx]
            self.params[idx] = self.bound_param(idx, val)

    def crossover(self, other):
        """keep some params and take some from  other"""
        if self.__class__ != other.__class__:
            raise Exception(f'crossover self class - {self.__class__}'
                            f' isnt same as other - {other.__class__}')

        num_of_params = len(self.params)
        param_index_other = random.sample(list(range(num_of_params)),
                                          random.randint(1, num_of_params))

        for i in param_index_other:
            self.params[i] = other.params[i]

    @classmethod
    @abstractmethod
    def create(cls):
        pass

    @abstractmethod
    def delta(self, idx):
        pass

    @abstractmethod
    def bound_param(self, idx, val):
        pass

    @abstractmethod
    def evaluate_state(self, game):
        pass

    @staticmethod
    def get_game_stats(game):
        return game.chips, game.turn, game.board

    @abstractmethod
    def __str__(self):
        pass


class CountChips(HeuFunctionInterface):

    def __init__(self, start_turn):
        super().__init__(start_turn)

    @classmethod
    def create(cls):
        start_turn = random.randint(30, 55)
        return CountChips(start_turn)

    def delta(self, idx):
        match idx:
            case 0:
                return random.choice([-3, -2, -1, 1, 2, 3])
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def bound_param(self, idx, val):
        match idx:
            case 0:
                return (val - 30) % 25 + 30
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def evaluate_state(self, game):
        stats = self.get_game_stats(game)
        return count_chips(stats,
                           start_turn=self.get_params()[0],
                           f=lambda turn: ((turn % 10) + 1))

    def __str__(self):
        return f'start_turn: {self.params[0]}'


class CountDangerEarlyGame(HeuFunctionInterface):

    @classmethod
    def create(cls):
        danger_mult = random.uniform(1.0, 10.0)
        end_turn = random.randint(5, 35)
        return CountDangerEarlyGame(danger_mult, end_turn)

    def delta(self, idx):
        match idx:
            case 0:
                return random.uniform(-2.5, 2.5)
            case 1:
                return random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def bound_param(self, idx, val):
        match idx:
            case 0:
                return (val - 1) % 9 + 1
            case 1:
                return (val - 5) % 30 + 5
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def evaluate_state(self, game):
        stats = self.get_game_stats(game)
        return count_danger_early_game(stats,
                                       end_turn=self.get_params()[1],
                                       f=lambda turn: (25 - turn) * self.get_params()[0])

    def __str__(self):
        return f'danger mult: {self.params[0]}, end_turn: {self.params[1]}'


class CountCorners(HeuFunctionInterface):
    def __init__(self, corner_divisor, corner_exponent, end_turn):

        super().__init__(corner_divisor, corner_exponent, end_turn)

    @classmethod
    def create(cls):
        corner_divisor = random.uniform(1.0, 10.0)
        corner_exponent = random.uniform(1.0, 3.0)
        end_turn = random.randint(15, 60)
        return CountCorners(corner_divisor, corner_exponent, end_turn)

    def delta(self, idx):
        match idx:
            case 0:
                return random.uniform(-2.5, 2.5)
            case 1:
                return random.uniform(-0.5, 0.5)
            case 2:
                return random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def bound_param(self, idx, val):
        match idx:
            case 0:
                return (val - 1) % 9 + 1
            case 1:
                return (val - 1) % 2 + 1
            case 2:
                return (val - 15) % 45 + 15
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def evaluate_state(self, game):
        stats = self.get_game_stats(game)
        params = self.get_params()
        return count_corners(stats,
                             end_turn=params[2],
                             f=lambda x: ((60 - x) / params[0]) ** params[1])

    def __str__(self):
        return f'corner divisor: {self.params[0]}, corner exponent: {self.params[1]}, end_turn: {self.params[2]}'


class CountSaferEarlyGame(HeuFunctionInterface):
    def __init__(self, safer_divisor, end_turn):
        super().__init__(safer_divisor, end_turn)

    @classmethod
    def create(cls):
        safer_divisor = random.uniform(1.0, 20.0)
        end_turn = random.randint(5, 35)
        return CountSaferEarlyGame(safer_divisor, end_turn)

    def delta(self, idx):
        match idx:
            case 0:
                return random.uniform(-4.5, 4.5)
            case 1:
                return random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def bound_param(self, idx, val):
        match idx:
            case 0:
                return (val - 1) % 19 + 1
            case 1:
                return (val - 5) % 30 + 5
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def evaluate_state(self, game):
        stats = self.get_game_stats(game)
        return count_safer(stats,
                           end_turn=self.get_params()[1],
                           f=lambda turn: (25 - turn) // self.get_params()[0])

    def __str__(self):
        return f'safer divisor: {self.params[0]}, end_turn: {self.params[1]}'


class MaximizeMyMoves(HeuFunctionInterface):
    def __init__(self, max_score, ratio):
        super().__init__(max_score, ratio)

    @classmethod
    def create(cls):
        max_score = random.uniform(10.0, 100.0)
        ratio = random.uniform(10.0, 20.0)
        return MaximizeMyMoves(max_score, ratio)

    def delta(self, idx):
        match idx:
            case 0:
                return random.uniform(-15.0, 15.0)
            case 1:
                return random.uniform(-2.0, 2.0)
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def bound_param(self, idx, val):
        match idx:
            case 0:
                return (val - 10) % 90 + 10
            case 1:
                return (val - 10) % 10 + 10
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def evaluate_state(self, game):
        params = self.get_params()
        return max_my_moves(game, params[0], lambda x: (x / params[1]))

    def __str__(self):
        return f'max my score: {self.params[0]}, max ratio div: {self.params[1]}'


class WeightedPieceCounter(HeuFunctionInterface):
    def __init__(self, max_turn):
        super().__init__(max_turn)

    @classmethod
    def create(cls):
        max_turn = random.randint(10, 60)
        return WeightedPieceCounter(max_turn)

    def delta(self, idx):
        match idx:
            case 0:
                return random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def bound_param(self, idx, val):
        match idx:
            case 0:
                return (val - 10) % 50 + 10
            case _:
                raise IndexError('index out of bounds for heu parameters!')

    def evaluate_state(self, game):
        stats = self.get_game_stats(game)
        params = self.get_params()
        return weighted_piece_counter(stats, max_turn=params[0])

    def __str__(self):
        return f'max turn: {self.params[0]}'
