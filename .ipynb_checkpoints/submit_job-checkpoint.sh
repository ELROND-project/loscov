#!/bin/bash

#SBATCH --job-name=all_covariance
#SBATCH --output=outputN=1e5s=0.05Nbina=5Nbin_z=5/parallel-job-output_%A_%a.log
#SBATCH --error=outputN=1e5s=0.05Nbina=5Nbin_z=5/parallel-job-error_%A_%a.log
#SBATCH --mail-user=daniel.johnson@umontpellier.fr
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --time=5-00:00:00
#SBATCH --array=0-131
#SBATCH --mem=10000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --dependency=singleton

source ~/lenstronomyenv/bin/activate

# Record start time
start_time=$(date +%s)

# Read task from task list
TASK_FILE="tasks.txt"
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ${TASK_FILE})

# Run the corresponding task
python -u job.py ${TASK}

# Record end time
end_time=$(date +%s)
runtime_seconds=$((end_time - start_time))
runtime_minutes=$((runtime_seconds / 60))

# Log runtime with task parameters safely
{
  flock -x 201
  echo "Job ${SLURM_ARRAY_TASK_ID} | Task: ${TASK} | Runtime: ${runtime_minutes} min (${runtime_seconds} sec)" >> runtime_N=1e5s=0.05Nbina=5Nbin_z=5.log
} 201>/tmp/runtime_N=1e5s=0.05Nbina=5Nbin_z=5.lock

cat outputN=1e5s=0.05Nbina=5Nbin_z=5/parallel-job-output_${SLURM_ARRAY_JOB_ID}_*.log > outputN=1e5s=0.05Nbina=5Nbin_z=5/combined_output.log
cat outputN=1e5s=0.05Nbina=5Nbin_z=5/parallel-job-error_${SLURM_ARRAY_JOB_ID}_*.log > outputN=1e5s=0.05Nbina=5Nbin_z=5/combined_errors.log
