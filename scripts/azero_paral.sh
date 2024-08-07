#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=cuda
#SBATCH --time=200:00:00
#SBATCH --job-name=az-64-final
#SBATCH --nodelist=n19
#SBATCH --output=azero_layer64_v4_%j.txt


#module load cuda/11.8
#module load python/miniconda3

# Detect cuda, cpu & gpu combo
python cpu-gpu-info.py

# Set the project directory
PROJECT_DIR="reversi-game"

# Save current directory
CURRENT_DIR=$(pwd)

# Change to the project directory
cd ../$PROJECT_DIR


# Activate your virtual environment
source ../venv310/bin/activate


# Ensure Python can find the modules in the project directory
export PYTHONPATH=$(pwd)


# Record the start time
start_time=$(date +%s)

# Run your Python script
#python algorithms/alphazero/AlphaZeroParallel.py "models_output/alpha-zero/FINAL/layer128-v1/model_4.pt" "models_output/alpha-zero/FINAL/layer128-v1/optimizer_4.pt"
#python algorithms/alphazero/AlphaZeroParallel.py
python algorithms/alphazero/AlphaZeroParallel.py "models_output/alpha-zero/FINAL/layer64-LAST-v3/model_3.pt" "models_output/alpha-zero/FINAL/layer64-LAST-v3/optimizer_3.pt"

# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$(($end_time - $start_time))

# Print the elapsed time
echo "Elapsed time: $(($elapsed_time / 3600))h $((($elapsed_time % 3600) / 60))m $(($elapsed_time % 60))s"


# Deactivate the virtual environment
deactivate