import warnings

warnings.filterwarnings('ignore', category=UserWarning)


from reversi_game.game_modes import (player_vs_ai_cli,
                                     player_vs_player_cli,
                                     ai_vs_ai_cli)
from reversi_game.read_all_agents import agents


def choose_agent(prompt):
    print(prompt)
    for i, agent in enumerate(agents):
        print(f"{i + 1}. {agent.name}")
    while True:
        choice = input("Choose an agent number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(agents):
            return agents[int(choice) - 1]
        print("Invalid choice, try again.")


def print_winner(name1, name2, game):
    win = 'None. Its draw!'
    if game.winner == 1:
        win = name1
    elif game.winner == 2:
        win = name2
    print(f'Winner is: {win} - {game.chips}')


def main():
    print("\nChoose game mode:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        name1 = input("\nEnter name for player #1: ").strip()
        name2 = input("\nEnter name for player #2: ").strip()
        game = player_vs_player_cli(name1, name2)
        print_winner(name1, name2, game)
    elif choice == "2":
        human = 'you'
        ai = choose_agent("\nChoose AI opponent:")

        player_turn = input("\nChoose your turn (1 = first, 2 = second): ").strip()
        player_turn = int(player_turn) if player_turn in ('1', '2') else 1

        min_move_time = input("\nSet minimum time for turn (0-5 sec): ").strip()
        min_move_time = float(min_move_time) if min_move_time.replace('.', '', 1).isdigit() else 0
        min_move_time = max(0, min(min_move_time, 5))  # clamp to 0–5

        game = player_vs_ai_cli(ai, player_turn=player_turn, min_move_time=min_move_time)
        if player_turn == 1:
            print_winner(human, ai.name, game)
        else:
            print_winner(ai.name, human, game)
    elif choice == "3":
        ai1 = choose_agent("\nChoose first AI:")
        ai2 = choose_agent("\nChoose second AI:")

        min_move_time = input("\nSet minimum time for turn (0-5 sec): ").strip()
        min_move_time = float(min_move_time) if min_move_time.replace('.', '', 1).isdigit() else 0
        min_move_time = max(0, min(min_move_time, 5))  # clamp to 0–5

        game = ai_vs_ai_cli(ai1, ai2, min_move_time=min_move_time)

        print_winner(ai1.name, ai2.name, game)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()

