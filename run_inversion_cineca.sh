#!/bin/bash
### NOTE
### This script is meant to run Backward Inference on the Cineca EuroHPC cluster.
###

#SBATCH --job-name=backward_inference         # Descriptive job name

#SBATCH --qos=normal                          # Quality of Service
#SBATCH --time=1-00:00:00                     # Maximum wall time for "normal" QOS

#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --gres=gpu:4                          # Number of GPUs per node
#SBATCH --ntasks-per-node=4                   # Tasks per node (one task per GPU)
#SBATCH --cpus-per-task=2                     # Number of CPU cores per task (adjust as needed)
#SBATCH --gpus-per-task=1                     # Number of GPUs per task
#SBATCH --ntasks=4                            # Total number of tasks
#SBATCH --partition=boost_usr_prod            # GPU-enabled partition
#SBATCH --output=%x-%j.SLURMout               # File for standard output (%x: job name, %j: job ID)
#SBATCH --error=%x-%j.SLURMerr                # File for standard error (%x: job name, %j: job ID)
#SBATCH --account=euhpc_d25_096               # Project account number


set -eo pipefail

# Parse arguments
CONFIG=""
CHECKPOINT=""
METHOD="pag-identity-embeddings" # Default method
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
        --method)
            METHOD="$2"
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
    echo "Usage: sbatch run_backward_inference_cineca.sh --config <config_name> --checkpoint <checkpoint_path> [--method <method_name>] [additional_args...]"
    exit 1
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required"
    echo "Usage: sbatch run_backward_inference_cineca.sh --config <config_name> --checkpoint <checkpoint_path> [--method <method_name>] [additional_args...]"
    exit 1
fi


echo "Running with the following arguments:"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Git commit: $(git rev-parse HEAD)"
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Method: $METHOD"
echo
echo "Additional arguments:"
printf '%s\n' "${ADDITIONAL_ARGS[@]}"
echo
echo "========================="
echo
echo

# Load necessary modules
module load profile/deeplrn

cd "/leonardo/home/userexternal/$USER/pag-llm"

module load python/3.11.7
# Only deactivate if a virtualenv is active; avoid printing "deactivate: command not found"
if type deactivate >/dev/null 2>&1; then
    deactivate
fi
source .venv/bin/activate

nvidia-smi

export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=donotsync
export WANDB_MODE=offline
export HF_EVALUATE_OFFLINE=1
# Reduce tqdm logging frequency to avoid filling SLURM logs; set minimum interval to 10s
export TQDM_MININTERVAL=30

for i in {0..3}; do
    srun \
      --exclusive \
      -n 1 \
      --gpus=1 \
      --gpu-bind=single:0 \
      --job-name="backward_gen_$i" \
    .venv/bin/python -u \
    run_backward_inference.py \
      --config-name "$CONFIG" \
      training.method="$METHOD" \
      +training.gpu_rank="$i" \
      +model.checkpoint_path="$CHECKPOINT" \
      "${ADDITIONAL_ARGS[@]}" \
      &
done

wait