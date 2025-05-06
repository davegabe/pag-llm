#!/bin/bash --login
### NOTE
### This script is meant to be run on the DI cluster.
###
### Change the job name and partition

# Job name:
#SBATCH --job-name=pag-llm_identity

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
#SBATCH --time=3-00:00:00

# Standard output and error to file
# %x: job name, %j: job ID
#SBATCH --output=%x-%j.SLURMout

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

cd "/home/$USER/pag-llm"
source .venv/bin/activate

nvidia-smi

export WANDB_API_KEY=donotsync
export WANDB_MODE=offline
srun python train.py "$@"

send_job_status_message finished
