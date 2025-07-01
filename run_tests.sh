#!/bin/bash
### NOTE
### This script is meant to run tests on the Cineca EuroHPC cluster.
### Usage: sbatch run_tests.sh --config <config_name> --checkpoint <checkpoint_path> [additional_args...]
###

#SBATCH --job-name=run_tests             # Descriptive job name

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

# Parse arguments
CONFIG=""
CHECKPOINT=""
ADDITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            ADDITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    echo "Usage: sbatch run_tests.sh --config <config_name> --checkpoint <checkpoint_path> [additional_args...]"
    exit 1
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required"
    echo "Usage: sbatch run_tests.sh --config <config_name> --checkpoint <checkpoint_path> [additional_args...]"
    exit 1
fi

echo "Running with the following arguments:"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo
echo "Additional arguments:"
printf '%s\n' "${ADDITIONAL_ARGS[@]}"
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

echo "Running test.py with config: $CONFIG and checkpoint: $CHECKPOINT"
srun python test.py --config "$CONFIG" --checkpoint "$CHECKPOINT" "${ADDITIONAL_ARGS[@]}"

echo "Running infer_backward_tinystories.py with config: $CONFIG and checkpoint: $CHECKPOINT"
srun python infer_backward_tinystories.py --config "$CONFIG" --checkpoint "$CHECKPOINT" "${ADDITIONAL_ARGS[@]}"
