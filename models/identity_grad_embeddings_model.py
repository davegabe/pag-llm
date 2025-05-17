import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm

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
        super().__init__('identity', model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.k_samples = 3
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

        # Use default normalization layer
        self.emb_norm = LlamaRMSNorm(
            self.model.config.hidden_size,
            eps=self.model.config.rms_norm_eps,
        )

    def _compute_losses(self, batch: BatchType, top_k_samples: int, tag: str) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Compute common losses used in both training and validation steps.

        Args:
            batch (BatchType): The batch of data.
            top_k_samples (int): Number of top-k samples to consider for accuracy.
            tag (str): Tag for logging.

        Returns:
            tuple: A tuple containing:
                - loss_ce: Cross-entropy loss
                - loss_grads: Gradient-based loss for first token
                - loss: Combined loss with lambda hyperparameters
                - top_k_accuracies: Top-k accuracies, as a dictionary to be logged
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

        # Get the embedding of the tokens (They have not been masked)
        valid_tokens = batch.input_ids[batch.attention_mask == 1]  # They must be flattened
        assert valid_tokens.ndim == 1, \
            f'Expected valid_tokens to be of shape (n\', ), but got {valid_tokens.shape}'

        # Calculate gradient-based loss if we're past the warmup period and lambda_loss_pag > 0
        if self.current_epoch >= self.warmup_pretrain_epochs and self.lambda_loss_pag > 0:
            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=create_graph)[0]
            # Take only the valid gradients
            grad_x_embed = grad_x_embed[batch.attention_mask == 1]

            n_first, vocab_size, d = valid_tokens.size(0), self.model.config.vocab_size, x_embed.size(-1)
            assert grad_x_embed.shape == (n_first, d), \
                f'Expected grad_x_embed to be of shape (n\'={n_first}, {d=}), but got {grad_x_embed.shape}'
            
            # Normalize the gradients based on the norm of the gradient
            grad_x_embed = grad_x_embed / grad_x_embed.norm(dim=-1, keepdim=True)

            # Dictionary to store gradient information for logging
            log_info = {}
            # Forward pass to get the logits and probabilities
            logits = forward_grad_embeddings(self.model, grad_x_embed, norm=self.emb_norm, log_info=log_info, tag=tag)
            assert logits.shape == (n_first, vocab_size), \
                f'Expected logits to be of shape (n\'={n_first}, {vocab_size=}), but got {logits.shape}'
                
            # Log the log_info dictionary
            self.log_dict(log_info, sync_dist=True)

            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.cross_entropy(
                input=logits,
                target=valid_tokens,
                reduction='mean'
            )

            # Compute the top-k accuracies
            top_k_accuracies = compute_top_k_accuracies(valid_tokens, logits, top_k_samples, tag)
        else:
            # We still need to return the loss and gradients for the first token
            loss_grads = torch.zeros_like(loss_ce)
            top_k_accuracies = dict()

        # Combine losses using the lambda hyperparameters
        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

        return loss_ce, loss_grads, loss, top_k_accuracies

    def _step(self, batch: BatchType, tag: str) -> torch.Tensor:
        """
        Compute the loss and perplexity on the forward pass.
        Compute also the accuracy of the Inverse First Token task.

        Args:
            batch (BatchType): The batch of data.
            tag (str): Tag for logging.
        """
        # Compute losses using common function
        loss_ce, loss_grads, loss, top_k_accuracies = self._compute_losses(
            batch,
            self.k_samples,
            tag,
        )

        if len(top_k_accuracies) > 0:
            # Log the top k accuracies
            self.log_dict(top_k_accuracies, sync_dist=True)

        # Calculate perplexity
        perplexity = torch.exp(loss_ce)
        self.log_dict({
            f'{tag}/loss_ce': loss_ce,
            f'{tag}/loss_first_inv': loss_grads,
            f'{tag}/loss': loss,
        }, prog_bar=True, sync_dist=True)
        self.log(f'{tag}/perplexity', perplexity, prog_bar=False, sync_dist=True)

        # Ensure that the model has no gradients
        self.model.zero_grad()

        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        with torch.inference_mode(mode=False):
            return self._step(batch, 'val')

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        """
        Compute ability to for Inverse Language Modeling task using masked sequences.
        Compute loss and accuracy on inverse K token task.

        Args:
            batch (BatchType): The batch of data.
            batch_idx (int): The index of the batch.
        """
        for first_tokens_to_predict in range(self.first_tokens_to_predict):
            for mask_value in self.mask_values:
                # Define the metric prefix
                exp_prefix = f'test/m_{mask_value}_t_{first_tokens_to_predict}'

                # Mask the first tokens
                input_ids = batch.input_ids.clone()
                input_ids[:, :first_tokens_to_predict] = mask_value
                batch.input_ids = input_ids

                with torch.inference_mode(mode=False):
                    # Compute losses using common function
                    loss_ce, loss_grads, loss, top_k_accuracies = self._compute_losses(
                        batch,
                        first_tokens_to_predict,
                        tag='test',
                    )

                    self.log_dict(top_k_accuracies, sync_dist=True, prog_bar=False)
                    
                    self.log_dict({
                        f'{exp_prefix}/loss_inv': loss_grads,
                    }, prog_bar=True, sync_dist=True)
        
        # Ensure that the model has no gradients
        self.model.zero_grad()
        return torch.tensor(0.0, device=self.device)
