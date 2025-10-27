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

def main():
    print("Choose game mode:")
    print("1. Human vs Human")
    print("2. Human vs AI")
    print("3. AI vs AI")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        name1 = input("Enter name for player #1: ").strip()
        name2 = input("Enter name for player #2: ").strip()
        player_vs_player_cli(name1, name2)
    elif choice == "2":
        ai = choose_agent("Choose AI opponent:")
        player_vs_ai_cli(ai)
    elif choice == "3":
        ai1 = choose_agent("Choose first AI:")
        ai2 = choose_agent("Choose second AI:")
        ai_vs_ai_cli(ai1, ai2)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

