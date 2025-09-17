# Base
sbatch run_inversion_cineca.sh --config base-small-offline --method base --checkpoint best-checkpoints/best-base.ckpt

# Identity
sbatch run_inversion_cineca.sh --config pag-identity-small-offline --method identity-grad --checkpoint best-checkpoints/best-identity-grad.ckpt
sbatch run_inversion_cineca.sh --config pag-identity-small-offline --method val-identity-grad --checkpoint best-checkpoints/best-val-identity-grad.ckpt

# Bert like
sbatch run_inversion_cineca.sh --config pag-identity-small-offline --method bert-like --checkpoint best-checkpoints/best-bert-like.ckpt
sbatch run_inversion_cineca.sh --config pag-identity-small-offline --method val-bert-like --checkpoint best-checkpoints/best-val-bert-like.ckpt

# Inv first
sbatch run_inversion_cineca.sh --config inv-first-small-offline --method inv-first --checkpoint best-checkpoints/best-inv-first.ckpt
sbatch run_inversion_cineca.sh --config inv-first-small-offline --method val-inv-first --checkpoint best-checkpoints/best-val-inv-first.ckpt


