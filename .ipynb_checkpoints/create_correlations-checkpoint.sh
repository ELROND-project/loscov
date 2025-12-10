#!/bin/bash

#SBATCH --job-name=preprocess
#SBATCH --output=job-output.log
#SBATCH --error=job-error.log
#SBATCH --ntasks=1
#SBATCH --mem=5000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --time=1-00:00:00  # Adjust if needed

source ~/lenstronomyenv/bin/activate

echo "Running preprocessing step..."
python -u correlations_and_distributions.py
