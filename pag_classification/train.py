from lightning import LightningModule, Trainer

from config import apply_config, SentenceClassificationConfig
from pag_classification.baseline_model import BaselineClassifier
from pag_classification.embeddings_datamodule import SentenceEmbeddingsDataModule
from pag_classification.pag_score_model import PagScoreClassifier


@apply_config('sentence_classification')
def main(cfg: SentenceClassificationConfig):
    model_name = cfg.config_to_train

    datamodule = SentenceEmbeddingsDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    model: LightningModule
    if model_name == 'baseline':
        model = BaselineClassifier(cfg)
    elif model_name == 'pag-score':
        model = PagScoreClassifier(cfg, datamodule.train_dataset)
    else:
        raise ValueError(f'Unknown model configuration to train {model_name}')

    print('Training:', type(model).__name__)


    training_output_dir = cfg.output_dir / f'training_{model_name}'
    training_output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(logger=True, max_epochs=cfg.epochs, default_root_dir=training_output_dir)
    trainer.fit(model, datamodule)

    trainer.test(model, datamodule)

    # Save this model as the new one
    final_ckpt_file = training_output_dir / f'{model_name}.ckpt'
    trainer.save_checkpoint(str(final_ckpt_file))
    print('Model saved to:', final_ckpt_file)


if __name__ == '__main__':
    main()
