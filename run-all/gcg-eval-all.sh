# Base
python eval_gcg.py --config-name base-small-offline training.method=base +model.checkpoint_path=best-checkpoints/best-base.ckpt

# Identity
python eval_gcg.py --config-name pag-identity-small-offline training.method=identity-grad +model.checkpoint_path=best-checkpoints/best-identity-grad.ckpt
python eval_gcg.py --config-name pag-identity-small-offline training.method=val-identity-grad +model.checkpoint_path=best-checkpoints/best-val-identity-grad.ckpt

# Bert like
python eval_gcg.py --config-name pag-identity-small-offline training.method=bert-like +model.checkpoint_path=best-checkpoints/best-bert-like.ckpt
python eval_gcg.py --config-name pag-identity-small-offline training.method=val-bert-like +model.checkpoint_path=best-checkpoints/best-val-bert-like.ckpt

# Inv first
python eval_gcg.py --config-name inv-first-small-offline training.method=inv-first +model.checkpoint_path=best-checkpoints/best-inv-first.ckpt
python eval_gcg.py --config-name inv-first-small-offline training.method=val-inv-first +model.checkpoint_path=best-checkpoints/best-val-inv-first.ckpt
