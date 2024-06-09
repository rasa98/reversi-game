import time
from game_logic import Othello
import random

# random.seed(time.time())


def player_vs_player_cli(name1, name2):
    game = Othello(players=(name1, name2))
    while not game.get_winner():
        print(f"Player turn: {game.white if game.player_turn == 1 else game.black}\n")
        print(game, "\n")
        print(f"Choose move:\n")
        moves_dict = dict(zip(range(1, 10 ** 6), list(game.valid_moves())))
        for i, field in moves_dict.items():
            print(f"{i} -> {field}")

        move = input()

        if move.isnumeric() and int(move) in moves_dict:
            num = int(move)
            if num in moves_dict:
                game.play_move(moves_dict[num])


def player_vs_ai_cli(player_name, ai: dict):
    ai_name, ai_first_turn, f = ai['name'], ai['first_turn'], ai['f']
    ai_turn = 1 if ai_first_turn else 2
    players = (ai_name, player_name) if ai_first_turn else (player_name, ai_name)

    game = Othello(players=players)
    while not game.get_winner():
        print(game, "\n")
        if game.player_turn == ai_turn:
            print("Ai turn\n")
            ai_move_choice = f(game.get_snapshot())
            game.play_move(ai_move_choice)
        else:
            print(f'Your turn')
            moves_dict = dict(zip(range(1, 10 ** 6), list(game.valid_moves())))
            for i, field in moves_dict.items():
                print(f"{i} -> {field}")

            move = input("Choose move:\n")

            if move.isnumeric() and int(move) in moves_dict:
                num = int(move)
                if num in moves_dict:
                    game.play_move(moves_dict[num])


def ai_vs_ai_cli(ai1, ai2, game=None):

    f1 = ai1.predict_best_move
    f2 = ai2.predict_best_move

    ai1_turn = 1

    players = (str(ai1), str(ai2))
    if game is None:
        game = Othello()
    game.players = players
    while game.get_winner() is None:
        if game.player_turn == ai1_turn:
            ai_moves, estimate = f1(game)
            ai_move_choice = random.choice(ai_moves)
            game.play_move(ai_move_choice)
        else:
            ai_moves, estimate = f2(game)
            ai_move_choice = random.choice(ai_moves)
            game.play_move(ai_move_choice)
    return game.get_winner()

