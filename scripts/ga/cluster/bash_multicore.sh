#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=all
#SBATCH --time=24:00:00
#SBATCH --job-name=rvrs-2
#SBATCH --nodelist=n10


# Activate your virtual environment
source ../../../venv/bin/activate

# Run your Python script
python ga_tuned_heu2.py

# Deactivate the virtual environment
deactivate