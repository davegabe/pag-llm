# Training hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1  # TODO: Check if this works correctly when > 1
NUM_EPOCHS = 3
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MAX_SEQ_LENGTH = 512

# Model configuration
MODEL_CHECKPOINT = "HuggingFaceTB/SmolLM2-135M"  # Pre-trained model
OUTPUT_DIR = "./checkpoints/finetuned-smollm2"

# Dataset configuration
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "validation"

# Logging and save configuration
LOGGING_STEPS = 100
EVALUATION_STEPS = 500
SAVE_STEPS = 1000
