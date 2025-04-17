import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel


class IdentityGradModel(BaseLMModel):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            config: LLMPagConfig
    ):
        super().__init__(model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.k_samples = 5

    def _forward_grad_embeddings(self, grad_x_embed: torch.Tensor) -> torch.Tensor:
        """
        Project the gradients of the embeddings to the vocabulary space using the head of the model.

        Args:
            grad_x_embed (torch.Tensor): The gradients of the embeddings. [batch_size, seq_len, embed_dim]

        Returns:
            tuple: A tuple containing:
                - logits: The logits of the model. [batch_size, seq_len, vocab_size]
                - probs: The probabilities of the model. [batch_size, seq_len, vocab_size]
                - top_k_indices: The indices of the top k tokens. [batch_size, seq_len, k]
        """
        assert grad_x_embed.ndim == 3
        n, t, d = grad_x_embed.shape
        v = self.model.config.vocab_size

        # Get the logits of the model
        logits = self.model.lm_head(grad_x_embed)
        assert logits.shape == (n, t, v)

        return logits

    def _compute_losses(self, batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute common losses used in both training and validation steps.

        Args:
            batch (BatchType): The batch of data.

        Returns:
            tuple: A tuple containing:
                - loss_ce: Cross-entropy loss
                - loss_grads: Gradient-based loss for first token
        """
        # Get the batch size and sequence length
        n, t = batch.input_ids.shape
        v = self.model.config.vocab_size

        if batch.input_ids.is_inference():
            # Clone inputs to avoid inference mode issues (caused by Lightning)
            input_ids = batch.input_ids.clone()
            attention_mask = batch.attention_mask.clone()
            shift_labels = batch.shift_labels.clone()
            # We don't need to create gradients for the validation step
            create_graph = False
        else:
            # We can use the original batch
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            shift_labels = batch.shift_labels
            # We need to create gradients for the training step
            create_graph = True

        # Get the embeddings of X
        x_embed = self.model.get_input_embeddings()(input_ids)
        d = x_embed.size(-1)
        assert x_embed.shape == (n, t, d)
        x_embed.requires_grad_(True)

        # Forward pass
        outputs: CausalLMOutputWithPast = self.model(
            inputs_embeds=x_embed,
            attention_mask=attention_mask,
            labels='dummy',
            shift_labels=shift_labels,
            output_hidden_states=False,
        )
        loss_ce = outputs.loss

        # Calculate gradient-based loss if we're past the warmup period
        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Get the gradients on all the tokens
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=create_graph)[0]

            # Forward pass to get the logits and probabilities
            grad_logits = self._forward_grad_embeddings(grad_x_embed)

            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.cross_entropy(
                input=grad_logits.view(n * t, v),
                target=input_ids.view(n * t),
                reduction='mean'
            )
        else:
            # We still need to return the loss and gradients for the first token
            loss_grads = torch.zeros_like(loss_ce)

        return loss_ce, loss_grads

    def training_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'train') -> torch.Tensor:
        # Compute losses using common function
        loss_ce, loss_grads = self._compute_losses(batch)

        # Combine losses
        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

        self.log_dict({
            f'{prefix_tag}/loss_ce': loss_ce,
            f'{prefix_tag}/loss_first_inv': loss_grads,
            f'{prefix_tag}/loss': loss,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'val') -> torch.Tensor:
        """
        Compute the validation loss and perplexity on the forward pass.
        Compute also the accuracy of the Inverse First Token task.

        Args:
            prefix_tag:
            batch (BatchType): The batch of data.
            batch_idx (int): The index of the batch.

        """
        # Set the model to evaluation mode
        self.model.eval()

        with torch.inference_mode(mode=False):
            # Compute losses using common function
            loss_ce, loss_grads = self._compute_losses(batch)

            # Combine losses
            loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

            # Calculate perplexity
            perplexity = torch.exp(loss_ce)
            self.log_dict({
                f'{prefix_tag}/loss_ce': loss_ce,
                f'{prefix_tag}/loss_first_inv': loss_grads,
                f'{prefix_tag}/perplexity': perplexity,
                f'{prefix_tag}/loss': loss,
            }, prog_bar=True)

        # Ensure that the model has no gradients
        self.model.zero_grad()

        return loss
