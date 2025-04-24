import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel
from models.common import forward_grad_embeddings, compute_top_k_accuracies


class IdentityGradEmbeddingsModel(BaseLMModel):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            config: LLMPagConfig,
    ):
        super().__init__(model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.k_samples = 3
        self.first_tokens_to_predict = 15
        self.mask_values = [
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id
        ]
        # Remove redundant tokens, if any (depending on the tokenizer)
        self.mask_values = list(set(self.mask_values))
        # Remove None values
        self.mask_values.remove(None)
        print(f"Test model with mask values: {self.mask_values}")

    def _compute_losses(self, batch: BatchType, first_tokens_to_predict: int = None) -> tuple:
        """
        Compute common losses used in both training and validation steps.

        Args:
            batch (BatchType): The batch of data.
            first_tokens_to_predict (int): Number of tokens to predict. 

        Returns:
            tuple: A tuple containing:
                - loss_ce: Cross-entropy loss
                - loss_grads: Gradient-based loss for first token
                - grad_x_embed: Gradients of embeddings
                - inv_first_label: Original first token labels
        """
        # If first_tokens_to_predict is None, use the default value
        if first_tokens_to_predict is None:
            first_tokens_to_predict = self.first_tokens_to_predict

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

        # Get the embedding of the first tokens (They have not been masked)
        inv_first_label = batch.input_ids[:, :first_tokens_to_predict].clone()
        tot_tokens, hidden_dim = input_ids.size(0) * first_tokens_to_predict, x_embed.size(-1)
        inv_first_label = inv_first_label.view(tot_tokens)  # They must be flattened

        # Calculate gradient-based loss if we're past the warmup period
        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=create_graph)[0]
            grad_x_embed = grad_x_embed[:, :first_tokens_to_predict, :]

            # Flatten all the gradients
            grad_x_embed = grad_x_embed.reshape(tot_tokens, hidden_dim)

            # Forward pass to get the logits and probabilities
            logits = forward_grad_embeddings(self.model, grad_x_embed)
            assert logits.shape == (tot_tokens, self.model.config.vocab_size)

            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.cross_entropy(
                input=logits,
                target=inv_first_label,
                reduction='mean'
            )
        else:
            # We still need to return the loss and gradients for the first token
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
                logits = forward_grad_embeddings(self.model, grad_x_embed)

                # Get the top k indices
                top_k_accuracies = compute_top_k_accuracies(inv_first_label, logits, self.k_samples, tag=prefix_tag)
                self.log_dict(top_k_accuracies, sync_dist=True)

            # Calculate perplexity
            perplexity = torch.exp(loss_ce)
            self.log_dict({
                'val/loss_ce': loss_ce,
                'val/loss_first_inv': loss_grads,
                'val/perplexity': perplexity,
                'val/loss': loss,
            }, prog_bar=True, sync_dist=True)

        # Ensure that the model has no gradients
        self.model.zero_grad()

        return loss

    def test_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'test') -> torch.Tensor:
        """
        Compute ability to for Inverse Language Modeling task using masked sequences.
        Compute loss and accuracy on inverse K token task.

        Args:
            prefix_tag: Tag prefix
            batch (BatchType): The batch of data.
            batch_idx (int): The index of the batch.
        """
        for first_tokens_to_predict in range(self.first_tokens_to_predict):
            for mask_value in self.mask_values:
                # Define the metric prefix
                exp_prefix = f'{prefix_tag}/m_{mask_value}_t_{first_tokens_to_predict}'

                # Mask the first tokens
                input_ids = batch.input_ids.clone()
                input_ids[:, :first_tokens_to_predict] = mask_value
                batch.input_ids = input_ids

                with torch.inference_mode(mode=False):
                    # Compute losses using common function
                    loss_ce, loss_grads, grad_x_embed, inv_first_label = self._compute_losses(
                        batch,
                        first_tokens_to_predict
                    )

                    # Combine losses
                    loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

                    if grad_x_embed is not None:
                        # Get the logits and probabilities
                        logits = forward_grad_embeddings(self.model, grad_x_embed)

                        # Get the top k indices
                        top_k_accuracies = compute_top_k_accuracies(
                            inv_first_label,
                            logits,
                            self.k_samples,
                            tag=exp_prefix
                        )
                        self.log_dict(top_k_accuracies, sync_dist=True)
                    
                    self.log_dict({
                        f'{exp_prefix}/loss_inv': loss_grads,
                    }, prog_bar=True, sync_dist=True)
        
        # Ensure that the model has no gradients
        self.model.zero_grad()
        return 0