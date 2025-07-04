import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel
from models.common import forward_grad_embeddings, compute_top_k_accuracies


class PosInvFirstTokenModel(BaseLMModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
            config: LLMPagConfig
    ):
        super().__init__('pos-inv-first', model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.k_samples = 5

    def _compute_losses(self, batch: BatchType) -> tuple:
        """
        Compute common losses used in both training and validation steps.
        
        Args:
            batch (BatchType): The batch of data.

        Returns:
            tuple: A tuple containing:
                - loss_ce: Cross-entropy loss
                - loss_grads: Gradient-based loss for first token
                - grad_x_embed: Gradients of embeddings
                - inv_first_label: Original first token labels
        """
        if batch.input_ids.is_inference():
            # Clone inputs to avoid inference mode issues (caused by Lightning)
            input_ids = batch.input_ids.clone()
            shift_labels = batch.shift_labels.clone()
            # We don't need to create gradients for the validation step
            create_graph = False
        else:
            # We can use the original batch
            input_ids = batch.input_ids
            shift_labels = batch.shift_labels
            # We need to create gradients for the training step
            create_graph = True

        attention_mask = batch.attention_mask

        # Mask the first token in input_ids
        input_ids[:, 0] = self.tokenizer.bos_token_id

        # Get the embeddings of X
        x_embed = self.model.get_input_embeddings()(input_ids)
        if create_graph:
            x_embed.requires_grad_(True)

        # Forward pass
        outputs: CausalLMOutputWithPast = self.model(
            inputs_embeds=x_embed,
            attention_mask=attention_mask,
            labels='dummy',
            shift_labels=shift_labels,
            output_hidden_states=False
        )
        loss_ce = outputs.loss

        # Calculate gradient-based loss if we're past the warmup period
        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Get the embedding of the first real token (not <s>)
            inv_first_label = batch.labels[:, 0].clone()

            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=create_graph)[0]

            # Forward pass to get the logits and probabilities
            new_embed = x_embed[:, 0, :] + grad_x_embed[:, 0, :]
            logits = forward_grad_embeddings(self.model, new_embed)
            
            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.cross_entropy(
                input=logits,
                target=inv_first_label,
                reduction='mean'
            )
        else:
            # We still need to return the loss and gradients for the first token
            inv_first_label = batch.labels[:, 0].clone()
            grad_x_embed = None
            loss_grads = torch.zeros_like(loss_ce)

        return loss_ce, loss_grads, grad_x_embed, inv_first_label

    def training_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'train') -> torch.Tensor:
        # Compute losses using common function
        loss_ce, loss_grads, _, _ = self._compute_losses(batch)
        
        # Combine losses
        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

        self.log_dict({
            f'{prefix_tag}/loss_ce': loss_ce,
            f'{prefix_tag}/loss_first_inv': loss_grads,
            f'{prefix_tag}/loss': loss,
        }, prog_bar=True, sync_dist=True)
        
        return loss


    def validation_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'val') -> torch.Tensor:
        """
        Compute the validation loss and perplexity on the forward pass.
        Compute also the accuracy of the Inverse First Token task.

        Args:
            prefix_tag: Tag prefix
            batch (BatchType): The batch of data.
            batch_idx (int): The index of the batch.
        """
        with torch.inference_mode(mode=False):
            # Compute losses using common function
            loss_ce, loss_grads, grad_x_embed, inv_first_label = self._compute_losses(batch)
            
            # Combine losses
            loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

            if grad_x_embed is not None:
                # Get the logits and probabilities
                logits = forward_grad_embeddings(self.model, grad_x_embed[:, 0, :])

                # Get the top k indices
                top_k_accuracies = compute_top_k_accuracies(inv_first_label, logits, self.k_samples, tag=prefix_tag)
                self.log_dict(top_k_accuracies, sync_dist=True)

            # Calculate perplexity
            perplexity = torch.exp(loss_ce)
            self.log_dict({
                f'{prefix_tag}/loss_ce': loss_ce,
                f'{prefix_tag}/loss_first_inv': loss_grads,
                f'{prefix_tag}/perplexity': perplexity,
                f'{prefix_tag}/loss': loss,
            }, prog_bar=True, sync_dist=True)

        # Ensure that the model has no gradients
        self.model.zero_grad()
        
        return loss
    
    def test_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'test') -> torch.Tensor:
        """
        Compute the test loss and perplexity on the forward pass.
        Compute also the accuracy of the Inverse First Token task.

        Args:
            prefix_tag: Tag prefix
            batch (BatchType): The batch of data.
            batch_idx (int): The index of the batch.
        """
        return self.validation_step(batch, batch_idx, prefix_tag)
