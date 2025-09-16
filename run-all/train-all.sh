# Base
sbatch run_job_cineca.sh --config-name base-small-offline training.method="base"

# Identity
sbatch run_job_cineca.sh --config-name pag-identity-small-offline training.method="identity-grad"
sbatch run_job_cineca.sh --config-name pag-identity-small-offline training.method="val-identity-grad"

# Bert like
sbatch run_job_cineca.sh --config-name pag-identity-small-offline training.method="bert-like"
sbatch run_job_cineca.sh --config-name pag-identity-small-offline training.method="val-bert-like"

# Inv first
sbatch run_job_cineca.sh --config-name pag-identity-small-offline training.method="inv-first"
sbatch run_job_cineca.sh --config-name pag-identity-small-offline training.method="val-inv-first"