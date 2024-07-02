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

The performance of the AI models have been evaluated and ranked using the Elo rating system. Below is the table showing the results:


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
