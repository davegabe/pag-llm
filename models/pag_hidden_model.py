from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .base_model import BaseLMModel
from config import Config
from utils.hdf5 import get_hidden_states_by_next_token


class PAGHiddenModel(BaseLMModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        config: Config,
        hdf5_file_path: str
    ):
        super().__init__(model, tokenizer, config)
        self.hdf5_file_path = hdf5_file_path
        self.hidden_layer_index = config.model.hidden_layer_index

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("PAGHiddenModel training is not implemented")
