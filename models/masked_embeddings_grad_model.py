import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel


class MaskedIdentityGradEmbeddingsModel(BaseLMModel):
    """
    This "masking" implies that, in a forward pass, a target token is replaced with [PAD] token.
    Then, the model is trained to predict the original token BUT on the gradients with respect to that token!

    In a nutshell:
    - usual LLM in a forward pass
    - BERT-like training on the gradients
    """

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
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs

    def _create_masked_input_ids(self, batch: BatchType) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create a masked input ids tensor where 10% of the tokens are replaced with [PAD] token.
        We use 10% instead of 15% as in BERT, because it is more difficult to train the model with a task.

        :param batch: BatchType, containing input_ids, attention_mask, and labels
        :return: Tuple of (masked_input_ids, masked_labels, mask)
        """
        sample_ratio = 0.1
        pad_token_id = self.tokenizer.pad_token_id
        masked_input_ids = batch.input_ids.clone()

        # Take 10% of the tokens in the input ids
        # Remember that input_ids is a batched tensor of shape (batch_size, sequence_length)
        assert masked_input_ids.ndim == 2, \
            f'Expected input_ids shape (batch_size, sequence_length), but got {masked_input_ids.shape}'

        n, t = masked_input_ids.shape
        assert n > 0, f'Expected batch size to be greater than 0, but got {n}'
        assert t > 0, f'Expected sequence length to be greater than 0, but got {t}'

        # Generate random scores for valid positions only
        rand_scores = torch.rand(n, t, device=masked_input_ids.device)
        rand_scores = rand_scores.masked_fill(batch.attention_mask == 0, 2.0)  # exclude padding with 2.0 (> 1.0)

        # Count valid tokens per row
        num_valid = batch.attention_mask.sum(dim=1)
        num_to_sample = torch.clamp((num_valid.float() * sample_ratio).long(), min=1)

        # Get sorted indices for each row (valid tokens come first due to masking)
        sorted_indices = rand_scores.argsort(dim=1)

        # Make a selection mask
        selection_mask = torch.zeros_like(masked_input_ids, dtype=torch.bool)
        for i in range(masked_input_ids.size(0)):
            k = num_to_sample[i]
            selection_mask[i, sorted_indices[i, :k]] = True

        masked_input_ids[selection_mask] = pad_token_id

        # Mask also the labels
        masked_shifted_labels = batch.shift_labels.clone()
        masked_shifted_labels[selection_mask] = pad_token_id

        return masked_input_ids, masked_shifted_labels, selection_mask.view(n, -1)

    def training_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'train') -> torch.Tensor:
        n, t = batch.input_ids.shape
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.labels is not None, f'Batch should contain the label for the loss function.'
        assert batch.labels.shape == batch.input_ids.shape

        # Get the embeddings of X
        x_embed = self.model.get_input_embeddings()(batch.input_ids)
        d = x_embed.size(-1)
        assert x_embed.shape == (n, t, d), f'Expected x_embed to be of shape (n, t, d), but got {x_embed.shape}'

        # Forward pass, with standard input X
        outputs: CausalLMOutputWithPast = self.model(inputs_embeds=x_embed,
                                                     attention_mask=batch.attention_mask,
                                                     labels=batch.labels,
                                                     shift_labels=batch.shift_labels,
                                                     output_hidden_states=False)

        loss_ce = outputs.loss

        # Get the embeddings of masked X
        masked_input_ids, masked_shifted_labels, mask = self._create_masked_input_ids(batch)
        assert masked_input_ids.shape == (n, t), \
            f'Expected masked_input_ids to be of shape (n, t), but got {masked_input_ids.shape}'
        assert mask.shape == (n, t), \
            f'Expected mask to be of shape (n, t), but got {mask.shape}'
        masked_x_embed = self.model.get_input_embeddings()(masked_input_ids)

        # Get the gradients on the masked embeddings
        masked_x_embed.requires_grad_(True)
        masked_x_embed.retain_grad()

        masked_outputs: CausalLMOutputWithPast = self.model(inputs_embeds=masked_x_embed,
                                                            attention_mask=batch.attention_mask,
                                                            labels='dummy',
                                                            shift_labels=masked_shifted_labels,
                                                            output_hidden_states=False)
        masked_x_grads = torch.autograd.grad(masked_outputs.loss, [masked_x_embed], create_graph=True)[0]

        # We want that gradients on the masked X will reconstruct the original X
        # ==> We want to ignore the gradients on non-masked/visible tokens
        padded_masked_x_grads = torch.zeros_like(masked_x_embed)
        padded_masked_x_grads[mask] = masked_x_grads[mask]
        padded_x_embed = torch.zeros_like(masked_x_embed)
        padded_x_embed[mask] = x_embed[mask]
        loss_grads = F.mse_loss(
            input=padded_masked_x_grads.view(n * t, d),
            target=padded_x_embed.view(n * t, d),
            reduction='sum',
        ) / mask.sum()

        if self.current_epoch < self.warmup_pretrain_epochs:
            # FIXME: after having understood the max VRAM usage, we can skip the PAG loss computation in the warmup phase
            loss_grads = 0.0


        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

        self.log_dict({
            f'{prefix_tag}/loss_ce': loss_ce,
            f'{prefix_tag}/loss_pag': loss_grads,
            f'{prefix_tag}/loss': loss,
        }, prog_bar=True)
        return loss
