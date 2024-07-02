# Reversi AI Project

## Description

This project involves developing an AI models for playing Reversi (also known as Othello). The goal was to create and train various AI models.
After training the models, they were benchmarked, and an Elo ranking system was used to evaluate their performance.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.6 or higher
- Required Python packages (listed in `requirements.txt`)

## Benchmarking Results - Elo Ranking After 100 Rounds

In this project, various AI models for the Reversi game were evaluated and ranked based on their Elo ratings after 100 rounds of testing. Below is a description of each model:

- **Alpha-MCTS - Depth 200:** An AlphaZero-style model utilizing Monte Carlo Tree Search (MCTS) with a search depth of 200. This model uses deep reinforcement learning to enhance its gameplay.

- **Alpha-MCTS - Depth 30:** An AlphaZero-style model using MCTS with a search depth of 30.

- **PPO-MLP:** A reinforcement learning model that uses Proximal Policy Optimization (PPO) with a Multi-Layer Perceptron (MLP) as its policy network.

- **MCTS Iteration Limit 500:** A model using Monte Carlo Tree Search (random simulations) with a limit of 500 iterations to explore possible game states.

- **PPO-CNN:** A PPO model with a Convolutional Neural Network (CNN) policy.

- **MCTS Iteration Limit 200:** A model using MCTS with a limit of 200 iterations.

- **MinMax with Human Set Heuristics:** A MinMax algorithm enhanced with heuristics manually set by a human for improved performance.

- **MinMax depth dyn GA2-best:** A MinMax algorithm optimized using a genetic algorithm (GA) to determine the best parameters.

- **MCTS Iteration Limit 30:** A model using MCTS with a limit of 30 iterations.

- **ARS (Augmented Random Search):** A reinforcement learning algorithm utilizing augmented random search techniques.

- **TRPO-CNN (Trust Region Policy Optimization):** A TRPO model using a CNN policy.

- **Random Model:** A baseline model that makes random moves.

Below is the table showing the results:


| Agent                          | Elo Rating |
|--------------------------------|------------|
| alpha-mcts - depth 200         | 1967.70    |
| alpha-mcts - depth 30          | 1682.13    |
| ppo_mlp                        | 1428.76    |
| Mcts iter_limit 500            | 1398.65    |
| ppo_cnn 19                     | 1268.30    |
| Mcts iter_limit 200            | 1257.60    |
| MinMax depth dyn GA2-best      | 1050.63    |
| MinMax human set               | 1035.34    |
| Mcts iter_limit 30             | 977.79     |
| ars 201                        | 964.15     |
| trpo_cnn 48                    | 779.77     |
| Random model                   | 633.45     |
