import time
from game_logic import Othello
import random

random.seed(time.time())


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


def ai_vs_ai_cli(ai1: dict, ai2: dict):
    ai1_name, ai1_first_turn, f1 = ai1['name'], ai1['first_turn'], ai1['f']
    ai2_name, ai2_first_turn, f2 = ai2['name'], ai2['first_turn'], ai2['f']

    ai1_turn = 1 if ai1_first_turn else 2

    players = (ai1_name, ai2_name)

    game = Othello(players=players)
    while game.get_winner() is None:
        if game.player_turn == ai1_turn:
            # print("Ai1 turn\n")
            ai_moves, estimate = f1(game.get_snapshot())
            # print('Same estimate moves: ', ai_moves)
            ai_move_choice = random.choice(ai_moves)
            # print(f'{ai1_name} chose field {ai_move_choice} with estimate: {estimate}')
            game.play_move(ai_move_choice)
        else:
            # print("Ai2 turn\n")
            ai_moves, estimate = f2(game.get_snapshot())

            ai_move_choice = random.choice(ai_moves)
            # print(f'{ai2_name} chose field {ai_move_choice} with estimate: {estimate}')
            game.play_move(ai_move_choice)
    game.get_winner_and_print()


if __name__ == '__main__':
    player_vs_player_cli("rasa", "ai")
