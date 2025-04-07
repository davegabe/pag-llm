import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel


class IdentityGradEmbeddingsModel(BaseLMModel):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            config: LLMPagConfig,
    ):
        super().__init__(model, tokenizer, config)
        self.hidden_layer_index = config.model.hidden_layer_index
        self.pag_classes = config.training.pag_classes  # Number of different next tokens to consider
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag

    def training_step(self, batch: BatchType, batch_idx: int):
        n, t = batch.input_ids.shape
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.labels is not None, f'Batch should contain the label for the loss function.'
        assert batch.labels.shape == batch.input_ids.shape

        # Get the embeddings
        x_embed = self.model.get_input_embeddings()(batch.input_ids)
        d = x_embed.size(-1)
        assert x_embed.shape == (n, t, d), f'Expected x_embed to be of shape (n, t, d), but got {x_embed.shape}'
        x_embed.requires_grad_(True)

        # Forward pass
        outputs: CausalLMOutputWithPast = self.model(inputs_embeds=x_embed,
                                                     attention_mask=batch.attention_mask,
                                                     labels=batch.labels,
                                                     output_hidden_states=False)

        loss_ce = outputs.loss

        grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=True)[0]
        loss_identity_grads = F.cosine_similarity(x_embed, grad_x_embed, dim=-1)
        assert loss_identity_grads.shape == (n, t)
        loss_identity_grads = 1 - loss_identity_grads.mean()

        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_identity_grads

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return loss
