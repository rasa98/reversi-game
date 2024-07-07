import random
import numpy as np
from itertools import permutations
from tqdm import trange
from game_modes import ai_vs_ai_cli
import os


class Player:
    def __init__(self, agent):
        self.agent = agent
        self.rating = 1200
        self.played_matches = 0

    def inc(self):
        self.played_matches += 1
    # def predict_best_move(self, game):
    #     return self.agent.predict_best_move(game)


class EloRating:
    def __init__(self, win=1, draw=0.5, loss=0):
        self.win = win
        self.draw = draw
        self.loss = loss

    @staticmethod
    def get_factor_k(pl):
        if pl.played_matches < 30:
            return 40
        return 10

    @staticmethod
    def expected_score(pl1, pl2):
        pl_1_expected = 1 / (1 + 10 ** ((pl2.rating - pl1.rating) / 400))
        pl_2_expected = 1 / (1 + 10 ** ((pl1.rating - pl2.rating) / 400))
        return pl_1_expected, pl_2_expected

    def calculate_new_rating(self, pl1, pl2, ac_score_1):
        ac_score_2 = 1 - ac_score_1
        ex_score_1, ex_score_2 = self.expected_score(pl1, pl2)
        pl1.rating = pl1.rating + self.get_factor_k(pl1) * (ac_score_1 - ex_score_1)
        pl2.rating = pl2.rating + self.get_factor_k(pl2) * (ac_score_2 - ex_score_2)


class Tournament:
    def __init__(self, agents, log_dir, rounds=100, save_nth=5, verbose=0):
        self.rounds = rounds
        self.log_dir = log_dir
        self.save_nth = save_nth
        self.players = [Player(agent) for agent in agents]
        self.elo = EloRating()
        self.verbose = verbose

    @staticmethod
    def get_actual_score(winner):
        """from perspective of first player"""
        if winner == 1:
            return 1
        elif winner == 2:
            return 0
        return 0.5  # its a draw

    def simulate(self):
        pairs = list(permutations(self.players, 2))
        for r in trange(self.rounds):
            random.shuffle(pairs)
            for pl1, pl2 in pairs:
                if self.verbose:
                    print(f'{pl1.agent} vs {pl2.agent}')
                winner = ai_vs_ai_cli(pl1.agent, pl2.agent)
                actual_score_1 = self.get_actual_score(winner)
                self.elo.calculate_new_rating(pl1, pl2, actual_score_1)
                pl1.inc()
                pl2.inc()
            
            if (r+1) % self.save_nth == 0:
                self.players.sort(key=lambda player: player.rating, reverse=True)
                self.save_simulation(r+1)

    def save_simulation(self, round_num):
        folder = 'elo output'
        os.makedirs(folder, exist_ok=True)
        with open(f'{folder}/{self.log_dir}_{round_num}.txt', 'w') as f:
            f.write(f'Elo ranking after {round_num} rounds:\n')
            for pl in self.players:
                f.write(f'\tAgent: {pl.agent}: {pl.rating}\n')
