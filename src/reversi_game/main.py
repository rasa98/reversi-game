from reversi_game.demo.cli.play_gui_cli import main as main_play
from reversi_game.demo.cli.benchmark_agents_cli import main as main_bench

def main():
    choice = input("Choose function: \n 1) - play game\n 2) - run benchmark with ai models\n").strip()
    if choice == "1":
        main_play()
    elif choice == "2":
        main_bench()
    else:
        print("Invalid choice. Defaulting to choice 1.")
        main_play()

if __name__ == "__main__":
    main()
