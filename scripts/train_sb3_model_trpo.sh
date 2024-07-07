#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=cuda
#SBATCH --time=100:00:00
#SBATCH --job-name=trpo-cnn
#SBATCH --nodelist=n01
#SBATCH --output=../reversi-game/scripts/rl/output/phase2/trpo/cnn/base_%j.txt


#module load cuda/11.8
#module load python/3

# Activate your virtual environment
source ../venv310/bin/activate

# Detect cuda, cpu & gpu combo
python3 cpu-gpu-info.py

# Set the project directory
PROJECT_DIR="reversi-game"

# Save current directory
CURRENT_DIR=$(pwd)

# Change to the project directory
cd ../$PROJECT_DIR


# Ensure Python can find the modules in the project directory
export PYTHONPATH=$(pwd)


# Record the start time
start_time=$(date +%s)

# Run your Python script
python3 scripts/rl/train_model_trpo.py

# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$(($end_time - $start_time))

# Print the elapsed time
echo "Elapsed time: $(($elapsed_time / 3600))h $((($elapsed_time % 3600) / 60))m $(($elapsed_time % 60))s"

# Deactivate the virtual environment
deactivate