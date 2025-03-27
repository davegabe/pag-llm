from .base_model import BaseLMModel
from utils.hdf5 import get_hidden_states_by_next_token

class PAGHiddenModel(BaseLMModel):
    def __init__(self, model, tokenizer, config, hdf5_file_path):
        super().__init__(model, tokenizer, config)
        self.hdf5_file_path = hdf5_file_path
        self.hidden_layer_index = config.model.hidden_layer_index

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("PAGHiddenModel training is not implemented")
