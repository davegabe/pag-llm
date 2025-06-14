import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig, CustomLLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel
from models.common import forward_grad_embeddings, compute_top_k_accuracies


class MultiInvFirstTokenModel(BaseLMModel):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            config: CustomLLMPagConfig | LLMPagConfig,
    ):
        super().__init__('multiinvfirst', model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.split_sentence_parts = 4  # Number of parts to split the sentence into
        self.k_samples = 5
        # self.backward_norm = nn.LayerNorm(config.model.hidden_size)

    def _compute_losses(self, batch: BatchType) -> tuple:
        """
        Compute common losses used in both training and validation steps.

        Args:
            batch (BatchType): The batch of data.

        Returns:
            tuple: A tuple containing:
                - loss_ce: Cross-entropy loss
                - loss_grads: Gradient-based loss for first token
                - inv_logits: Predicted logits for the first token, in various splits
                - inv_labels: Original first token labels, in various splits
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

        ## FORWARD MODE
        # Compute the usual CE loss
        forward_outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels='dummy',
            shift_labels=shift_labels,
            output_hidden_states=False
        )
        forward_loss_ce = forward_outputs.loss

        ## BACKWARD MODE
        # Calculate gradient-based loss if we're past the warmup period
        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Split the sentence into multiple parts
            # Choose K indexes to split the sentence into parts
            all_indices = torch.ones(batch.input_ids.size(-1), device=input_ids.device, dtype=torch.float32)
            split_indexes = torch.multinomial(all_indices, self.split_sentence_parts, replacement=False).sort()[
                0].long()

            x_embed = self.model.get_input_embeddings()(input_ids)
            pad_embed = self.model.get_input_embeddings()(
                torch.tensor(self.tokenizer.pad_token_id, device=input_ids.device)
            )

            all_loss_grads = []
            all_backward_logits = []
            all_backward_labels = []

            for split_i in split_indexes:
                # Take from split_i to the end of the sentence
                inputs_embeds_split = x_embed[:, split_i:, :].contiguous().clone()
                shift_labels_split = shift_labels[:, split_i:].contiguous()
                attention_mask_split = attention_mask[:, split_i:].contiguous()

                # Mask the first token in input token embeddings
                inputs_embeds_split[:, 0] = pad_embed

                inputs_embeds_split = inputs_embeds_split.contiguous().detach()
                inputs_embeds_split.requires_grad_(create_graph)

                # Forward pass
                outputs: CausalLMOutputWithPast = self.model(
                    inputs_embeds=inputs_embeds_split,
                    attention_mask=attention_mask_split,
                    labels='dummy',
                    shift_labels=shift_labels_split,
                    output_hidden_states=False
                )
                split_loss_ce = outputs.loss

                inv_first_label = shift_labels_split[:, 0].clone()
                # Get the gradients on the first token
                grad_x_embed = torch.autograd.grad(split_loss_ce, [inputs_embeds_split], create_graph=create_graph)[0]

                # Forward pass to get the logits and probabilities
                logits = forward_grad_embeddings(self.model, grad_x_embed[:, 0, :])

                # We want that gradients on the first token will reconstruct the original token
                loss_grads = F.cross_entropy(
                    input=logits,
                    target=inv_first_label,
                    reduction='mean'
                )
                all_loss_grads.append(loss_grads)
                all_backward_logits.append(logits)
                all_backward_labels.append(inv_first_label)

            loss_grads = torch.stack(all_loss_grads).mean() if all_loss_grads else torch.zeros_like(forward_loss_ce)
            inv_logits = torch.stack(all_backward_logits) if all_backward_logits else None
            inv_labels = torch.stack(all_backward_labels) if all_backward_labels else None

        else:
            # If we are in the warmup period, we do not compute the backward loss
            loss_grads = torch.zeros_like(forward_loss_ce)
            inv_logits = None
            inv_labels = None

        return forward_loss_ce, loss_grads, inv_logits, inv_labels

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
            loss_ce, loss_grads, inv_logits, inv_labels = self._compute_losses(batch)

            # Combine losses
            loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

            if inv_logits is not None:
                # Get the top k indices
                vocab_size = inv_logits.size(-1)
                inv_logits, inv_labels = inv_logits.view(-1, vocab_size), inv_labels.view(-1)
                top_k_accuracies = compute_top_k_accuracies(inv_labels, inv_logits, self.k_samples, tag=prefix_tag)
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
