import math
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


from reversi_game.bench_agent import (benchmark,
                         bench_both_sides)
from reversi_game.read_all_agents import agents


def choose_agent(prompt):
    print('\n'+prompt)
    for i, agent in enumerate(agents):
        print(f"{i + 1}. {agent.name}")
    while True:
        choice = input("Choose an agent number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(agents):
            return agents[int(choice) - 1]
        print("\nInvalid choice, try again.")

def get_benchmark_settings():
    # number of matches
    matches = input("\nNumber of matches (1-1000): ").strip()
    matches = int(matches) if matches.isdigit() else 1
    matches = max(1, min(matches, 1000))  # clamp 1–1000


    # double matches (both play first)
    double = input("\nRun double matches (both play first)? (y/n): ").strip().lower() == 'y'

    return matches, double




def main():
    print(f'\nBenchmarking ai agent.')
    ai1 = choose_agent("Choose AI #1:")
    ai2 = choose_agent("Choose AI #2:")

    matches, double = get_benchmark_settings()
    print(f"\nRunning benchmark: {matches} matches, both_sides={double}\n")
    if double:
        bench_both_sides(ai1, ai2, times=matches, verbose=1)
    else:
        benchmark(ai1, ai2, times=matches, verbose=1)


if __name__ == "__main__":
    main()