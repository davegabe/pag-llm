#!/bin/bash --login
set -eo pipefail
### NOTE
### This script is meant to be run on the DI cluster.
###
### Change the job name and partition

# Job name:
#SBATCH --job-name=pag-llm_tinystories_bert

# Define the partition on which the job shall run
#SBATCH --partition=department_only

# Number of processes.
# Unless programmed using MPI,
# most programs using GPU-offloading only need
# a single CPU-based process to manage the device(s)
#SBATCH --ntasks=1

# Type and number of GPUs
# The type is optional.
#SBATCH --gpus=1

# Total CPU memory
# All available memory per GPU is allocated by default.
# Specify "M" or "G" for MB and GB respectively
#SBATCH --mem=8G

# Wall time
# Format: "minutes", "hours:minutes:seconds",
# "days-hours", or "days-hours:minutes"
#SBATCH --time=07:00:00

# Standard output and error to file
# %x: job name, %j: job ID
#SBATCH --output=%x-%j.SLURMout

echo "Running with the following arguments:"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo "========================="
echo
echo

cd "/home/$USER/pag-llm"
source .venv/bin/activate

export WANDB_MODE=offline
srun python train.py
