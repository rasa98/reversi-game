#!/bin/bash

#SBATCH --nodes=3
#SBATCH --ntasks-per-node=12
#SBATCH --partition=all
#SBATCH --time=00:10:00
#SBATCH --job-name=rvrs-2

cd ../../

# Activate your virtual environment
source ../venv/bin/activate


# Run your Python script
python ga_tuned_heu2_spark.py 18 "heuristics/ga/ga_cluster/dist_spark"

# Deactivate the virtual environment
deactivate