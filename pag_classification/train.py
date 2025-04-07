import pathlib
from typing import Generator

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from config import apply_config, SentenceClassificationConfig
from pag_classification.baseline_model import BaselineClassifier
from pag_classification.embeddings_datamodule import SentenceEmbeddingsDataModule
from pag_classification.pag_identity_model import PagIdentityClassifier
from pag_classification.pag_score_model import PagScoreSimilarSamplesClassifier, PagScoreSimilarFeaturesClassifier


@apply_config('sentence_classification')
def main(cfg: SentenceClassificationConfig):
    print('Loading with config:', cfg)

    model_name = cfg.config_to_train
    output_name = cfg.run_name or model_name

    datamodule = SentenceEmbeddingsDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    model: LightningModule
    if model_name == 'baseline':
        model = BaselineClassifier(cfg)
    elif model_name == 'pag-score-similar-samples':
        model = PagScoreSimilarSamplesClassifier(cfg, datamodule.train_dataset)
    elif model_name == 'pag-score-similar-features':
        model = PagScoreSimilarFeaturesClassifier(cfg, datamodule.train_dataset)
    elif model_name == 'pag-identity':
        model = PagIdentityClassifier(cfg)
    else:
        raise ValueError(f'Unknown model configuration to train {model_name} (output: {output_name})')

    print('Training:', type(model).__name__)

    training_output_dir = cfg.output_dir / f'training_{output_name}'
    training_output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(dirpath=training_output_dir,
                                          save_top_k=1,
                                          monitor="val/loss",
                                          filename=output_name)

    wandb_logger = WandbLogger(entity='pag-llm-team',
                               project='sentence-classification',
                               name=output_name,
                               log_model=True)
    trainer = Trainer(logger=wandb_logger,
                      max_epochs=cfg.epochs,
                      default_root_dir=None if cfg.resume_training else training_output_dir,
                      callbacks=[checkpoint_callback])

    last_train_ckpt_file: pathlib.Path | None = None
    if cfg.resume_training:
        print('Resuming old training state')
        # Find the last version
        lightning_logs_dir = training_output_dir / 'lightning_logs'
        lightning_version_dirs: Generator[pathlib.Path, None, None] = lightning_logs_dir.iterdir()
        last_version_num = max(int(version_name.name[8:]) for version_name in lightning_version_dirs)
        last_checkpoints_dir = lightning_logs_dir / f'version_{last_version_num}' / 'checkpoints'
        last_train_ckpt_file = max(last_checkpoints_dir.iterdir(), key=lambda x: x.stat().st_ctime)
        print('Last training checkpoint:', last_train_ckpt_file)

    trainer.fit(model, datamodule, ckpt_path=last_train_ckpt_file)

    trainer.test(model, datamodule)


if __name__ == '__main__':
    main()
