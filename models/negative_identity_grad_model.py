from transformers import PreTrainedModel, PreTrainedTokenizerFast

from config import LLMPagConfig
from models.identity_grad_embeddings_model import IdentityGradEmbeddingsModel


class NegativeIdentityGradEmbeddingsModel(IdentityGradEmbeddingsModel):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            config: LLMPagConfig,
    ):
        super().__init__(model, tokenizer, config, grad_logits_sign=-1)
