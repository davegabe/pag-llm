#!/bin/bash
### NOTE
### This script is meant to be run on the Cineca EuroHPC cluster.
###

#SBATCH --job-name=inverse_lm                 # Descriptive job name

#SBATCH --qos=normal                          # Quality of Service
#SBATCH --time=1-00:00:00                     # Maximum wall time for "normal" QOS

#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=4                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:4                          # Number of GPUs per node
#SBATCH --partition=boost_usr_prod            # GPU-enabled partition
#SBATCH --output=%x-%j.SLURMout               # File for standard output (%x: job name, %j: job ID)
#SBATCH --error=%x-%j.SLURMerr                # File for standard error (%x: job name, %j: job ID)
#SBATCH --account=euhpc_d25_096               # Project account number


set -eo pipefail


echo "Running with the following arguments:"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo
echo "User specified arguments:"
printf '%s\n' "$@"
echo
echo "========================="
echo
echo

# Load necessary modules
module load profile/deeplrn
module load anaconda3
activate ilm

cd "/leonardo/home/userexternal/$USER/pag-llm"

nvidia-smi

export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=donotsync
export WANDB_MODE=offline
srun python train.py "$@"
