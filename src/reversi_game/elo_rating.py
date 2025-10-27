import os, random
from itertools import permutations
from tqdm import trange
from collections import defaultdict

from reversi_game.game_modes import ai_vs_ai_train


class Player:
    def __init__(self, agent):
        self.agent = agent
        self.rating = 1200
        self.played_matches = 0

    def inc(self):
        self.played_matches += 1

    def __repr__(self):
        return str(self.agent)


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
    def __init__(self, agents, log_filename, log_dir='elo_output', rounds=100, save_nth=5, verbose=0, banned=None):
        self.rounds = rounds
        self.log_filename = log_filename
        self.log_dir = log_dir
        self.save_nth = save_nth
        self.players = [Player(agent) for agent in agents]
        self.elo = EloRating()
        self.verbose = verbose
        self.banned = set() if banned is None else banned

        self.stats = defaultdict(lambda: [0, 0, 0])

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
                if (pl1.agent, pl2.agent) in self.banned or (pl2.agent, pl1.agent) in self.banned:
                    continue
                if self.verbose:
                    print(f'{pl1.agent} vs {pl2.agent}')
                game = ai_vs_ai_train(pl1.agent, pl2.agent)
                winner = game.get_winner()
                self.update_stats(pl1, pl2, winner)
                actual_score_1 = self.get_actual_score(winner)
                self.elo.calculate_new_rating(pl1, pl2, actual_score_1)
                pl1.inc()
                pl2.inc()
            
            if (r+1) % self.save_nth == 0:
                self.players.sort(key=lambda player: player.rating, reverse=True)
                self.save_simulation(r+1)

    def update_stats(self, pl1, pl2, winner):
        if winner == 1:
            self.stats[(str(pl1), str(pl2))][0] += 1
        if winner == 2:
            self.stats[(str(pl1), str(pl2))][2] += 1
        if winner == 0:
            self.stats[(str(pl1), str(pl2))][1] += 1

    def save_simulation(self, round_num):
        os.makedirs(self.log_dir, exist_ok=True)
        output_path = os.path.join(self.log_dir, f'{self.log_filename}_{round_num}.txt')
        with open(output_path, 'w') as f:
            f.write(f'Elo ranking after {round_num} rounds:\n')
            for pl in self.players:
                f.write(f'\tAgent: {pl.agent}: {pl.rating}\n')

            f.write('\nStats:\n')
            f.write('(player 1, player 2): [win, draw, loss]\n\n')
            sorted_data = sorted(self.stats.items(), key=lambda item: (item[0][0], item[1][0]))
            for key, value in sorted_data:
                line = f"{key}: {value}\n"
                f.write(line)


