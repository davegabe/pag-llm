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
#SBATCH --account=EUHPC_D25_096               # Project account number


send_job_status_message() {
	TELEGRAM_BOT_TOKEN="7587099540:AAFaYgws0ROIjDLtbaTuBk5zQbeMHejrDrs"
	TELEGRAM_DEST_ID="-1002670256273"
	curl -X POST \
	     --silent \
	     -H 'Content-Type: application/json' \
	     -d "{\"chat_id\": \"$TELEGRAM_DEST_ID\", \"text\": \"Job $1: $SLURM_JOB_NAME, with ID: $SLURM_JOB_ID, triggered by $USER\"}" \
	     https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage \
	     >/dev/null
}

trap 'send_job_status_message ERRORED' ERR

set -eo pipefail

send_job_status_message started

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
module load cineca-ai/4.3.0

cd "/home/$USER/pag-llm"
source .venv/bin/activate

nvidia-smi

export WANDB_API_KEY=donotsync
export WANDB_MODE=offline
srun python train.py "$@"

send_job_status_message finished
