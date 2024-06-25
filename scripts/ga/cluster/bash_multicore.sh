#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=cuda
#SBATCH --time=100:00:00
#SBATCH --job-name=ga
#SBATCH --nodelist=n19
#SBATCH --output=../project/reversi-game/models_output/ga/output/ga_%j.txt


# Activate your virtual environment
source ../project/venv310/bin/activate

# Detect cuda, cpu & gpu combo
python cpu-gpu-info.py

# Change to the project directory
cd ../project/reversi-game/


# Ensure Python can find the modules in the project directory
export PYTHONPATH=$(pwd)


# Record the start time
start_time=$(date +%s)

# Run your Python script
python scripts/ga/cluster/ga_tuned_heu2.py

# Record the end time
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$(($end_time - $start_time))

# Print the elapsed time
echo "Elapsed time: $(($elapsed_time / 3600))h $((($elapsed_time % 3600) / 60))m $(($elapsed_time % 60))s"


# Deactivate the virtual environment
deactivate
