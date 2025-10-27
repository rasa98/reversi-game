import random, string, time

from reversi_game.game_logic import Othello


def coord_to_label(row, col):
    return f"{string.ascii_uppercase[row]}{col + 1}"


def player_vs_player_cli(name1, name2):
    game = Othello(players=(name1, name2))
    print(game, "\n")
    while not game.get_winner():
        print(f"Player turn: {game.white if game.player_turn == 1 else game.black}\n")
        cli_player_move(game)
        print(game, "\n")
    return game



def player_vs_ai_cli(ai, player_turn=1, min_move_time=0):
    ai_name, f = (ai.name, ai.predict_best_move)
    player_name = 'you'
    players = (ai_name, player_name) if player_turn == 2 else (player_name, ai_name)

    game = Othello(players=players)
    print('\n', game)
    while not game.get_winner():

        if game.player_turn != player_turn:
            print(f"{ai.name} turn\n")
            start_time = time.perf_counter()

            ai_move_choice = f(game.get_snapshot())
            ai_move_choice = random.choice(ai_move_choice[0])  # need this, cuz ai agent gives different output...
            game.play_move(ai_move_choice)

            end_time = time.perf_counter()
            delta_time = end_time - start_time
            if delta_time < min_move_time:
                time.sleep(min_move_time - delta_time)
        else:
            print(f'Your turn')
            cli_player_move(game)

        print('\n', game)


    return game


def cli_player_move(game):
    moves_dict = dict(zip(range(1, 10 ** 6), list(game.valid_moves())))
    for i, field in moves_dict.items():
        print(f"{i} -> {coord_to_label(*field)}")
    move = input("Choose move:\n")
    if move.isnumeric() and int(move) in moves_dict:
        num = int(move)
        if num in moves_dict:
            game.play_move(moves_dict[num])


def ai_vs_ai_train(ai1, ai2, game=None):
    #  for training models, without print....
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

    return game


def ai_vs_ai_cli(ai1, ai2, game=None, min_move_time=0):
    f1 = ai1.predict_best_move
    f2 = ai2.predict_best_move

    ai1_turn = 1

    players = (str(ai1), str(ai2))
    if game is None:
        game = Othello()
    game.players = players
    print('\n', game)
    while game.get_winner() is None:
        start_time = time.perf_counter()
        if game.player_turn == ai1_turn:
            ai_moves, estimate = f1(game)
            ai_move_choice = random.choice(ai_moves)
            game.play_move(ai_move_choice)
        else:
            ai_moves, estimate = f2(game)
            ai_move_choice = random.choice(ai_moves)
            game.play_move(ai_move_choice)

        end_time = time.perf_counter()
        delta_time = end_time - start_time
        if delta_time < min_move_time:
            time.sleep(min_move_time - delta_time)

        print('\n', game)

    return game


