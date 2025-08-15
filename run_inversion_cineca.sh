#!/bin/bash
### NOTE
### This script is meant to run ILM Inversion evaluation on the Cineca EuroHPC cluster.
###

#SBATCH --job-name=inverse_lm_inf             # Descriptive job name

#SBATCH --qos=normal                          # Quality of Service
#SBATCH --time=1-00:00:00                     # Maximum wall time for "normal" QOS

#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --gres=gpu:1                          # Number of GPUs per node
#SBATCH --ntasks-per-node=1                   # Number of MPI tasks per node (Same number as the GPUs)
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task (adjust as needed)
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

cd "/leonardo/home/userexternal/$USER/pag-llm"

module load python/3.11.7
deactivate || true
source .venv/bin/activate

nvidia-smi

export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=donotsync
export WANDB_MODE=offline
export HF_EVALUATE_OFFLINE=1

echo "===== Starting Inversion Execution =====" | tee /dev/stderr
srun python infer_backward_tinystories.py "$@"
echo
echo "===== Starting Inversion Evaluation Statistics =====" | tee /dev/stderr
srun python inverse_lm_stats.py "$@"