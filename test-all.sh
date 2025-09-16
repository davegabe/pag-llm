# PYTHON_SCRIPT=test.py
PYTHON_SCRIPT=reformat_gcg_attacks.py

# Base
python $PYTHON_SCRIPT --config-name base-small-offline +model.checkpoint_path=best-checkpoints/best-base.ckpt

# Bert like
python $PYTHON_SCRIPT --config-name pag-identity-small-offline +model.checkpoint_path=best-checkpoints/best-bert-like.ckpt training.method=bert-like
python $PYTHON_SCRIPT --config-name pag-identity-small-offline +model.checkpoint_path=best-checkpoints/best-grad-bert-like.ckpt training.method=val-bert-like

# Inv first
python $PYTHON_SCRIPT --config-name inv-first-small-offline +model.checkpoint_path=best-checkpoints/best-inv-first.ckpt training.method=inv-first
python $PYTHON_SCRIPT --config-name inv-first-small-offline +model.checkpoint_path=best-checkpoints/best-grad-inv-first.ckpt training.method=val-inv-first

# Pag identity
python $PYTHON_SCRIPT --config-name pag-identity-small-offline +model.checkpoint_path=best-checkpoints/best-pag-identity.ckpt training.method=identity-grad
python $PYTHON_SCRIPT --config-name pag-identity-small-offline +model.checkpoint_path=best-checkpoints/best-grad-identity.ckpt training.method=val-identity-grad
