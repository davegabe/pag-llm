import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel
from models.common import compute_top_k_accuracies, forward_grad_embeddings


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
        super().__init__('bertlike', model, tokenizer, config)
        self.hidden_layer_index = config.model.hidden_layer_index
        self.pag_classes = config.training.pag_classes  # Number of different next tokens to consider
        self.criterion = torch.nn.CrossEntropyLoss()
        self.k_samples = 3
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs

    @torch.no_grad()
    def _create_masked_input_ids(self, batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a masked input ids tensor where 10% of the tokens are replaced with [PAD] token.
        We use 10% instead of 15% as in BERT, because it is more difficult to train the model with a task.

        :param batch: BatchType, containing input_ids, attention_mask, and labels
        :return: Tuple of (masked_input_ids, mask)
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

        # We do not need to mask the labels.
        # Suppose we mask token i-th,
        # then the label for token i-th is at position shift_labels[i-1].
        # Since the label is in the past, it won't influence the gradients on token i-th.
        # This has been tested doing .logits[0][i-1].sum().backward(),
        # and asserting that inputs_embeds.grad[0][i] == 0.

        return masked_input_ids, selection_mask.view(n, -1)

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
        n, t = batch.input_ids.shape
        assert batch.attention_mask.shape == batch.input_ids.shape
        assert batch.labels is not None, f'Batch should contain the label for the loss function.'
        assert batch.labels.shape == batch.input_ids.shape

        if batch.input_ids.is_inference():
            input_ids = batch.input_ids.clone()
            shift_labels = batch.shift_labels.clone()
            # We need to create gradients for the training step
            create_graph = False
        else:
            # We can use the original batch
            input_ids = batch.input_ids
            shift_labels = batch.shift_labels
            # We need to create gradients for the training step
            create_graph = True


        # Get the embeddings of X
        x_embed = self.model.get_input_embeddings()(input_ids)
        d = x_embed.size(-1)
        assert x_embed.shape == (n, t, d), f'Expected x_embed to be of shape (n, t, d), but got {x_embed.shape}'

        # Forward pass, with standard input X
        outputs: CausalLMOutputWithPast = self.model(inputs_embeds=x_embed,
                                                     attention_mask=batch.attention_mask,
                                                     labels='dummy',
                                                     shift_labels=shift_labels,
                                                     output_hidden_states=False)

        loss_ce = outputs.loss

        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Get the embeddings of masked X
            masked_input_ids, mask = self._create_masked_input_ids(batch)
            assert masked_input_ids.shape == (n, t), \
                f'Expected masked_input_ids to be of shape (n, t), but got {masked_input_ids.shape}'
            assert mask.shape == (n, t), \
                f'Expected mask to be of shape (n, t), but got {mask.shape}'
            masked_x_embed = self.model.get_input_embeddings()(masked_input_ids)

            # Get the gradients on the masked embeddings
            if create_graph:
                masked_x_embed.requires_grad_(True)

            masked_outputs: CausalLMOutputWithPast = self.model(inputs_embeds=masked_x_embed,
                                                                attention_mask=batch.attention_mask,
                                                                labels='dummy',
                                                                shift_labels=shift_labels,
                                                                output_hidden_states=False)
            masked_x_grads = torch.autograd.grad(
                masked_outputs.loss,
                [masked_x_embed],
                create_graph=create_graph
            )[0]

            # We want that gradients on the masked X will reconstruct the original X
            # ==> We want to ignore the gradients on non-masked/visible tokens
            valid_masked_x_grads = masked_x_grads[mask]
            target_input_ids = input_ids[mask]
            assert valid_masked_x_grads.ndim == 2
            assert valid_masked_x_grads.size(1) == d
            assert target_input_ids.ndim == 1
            assert target_input_ids.size(0) == valid_masked_x_grads.size(0)

            # Forward pass to get the logits and probabilities
            # Since we are using the gradients, we need to add the tokens we computed the gradients for ([PAD] token)
            valid_masked_x_grads = self.tokenizer.pad_token_id - valid_masked_x_grads
            valid_input_ids_predicted = forward_grad_embeddings(
                self.model,
                valid_masked_x_grads,
            )

            loss_grads = F.cross_entropy(
                input=valid_input_ids_predicted,
                target=target_input_ids,
                reduction='mean',
            )

            # Compute the top-k accuracies
            top_k_accuracies = compute_top_k_accuracies(target_input_ids, valid_input_ids_predicted, top_k_samples, tag)
        else:
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
            'val/loss_ce': loss_ce,
            'val/loss_mask': loss_grads,
            'val/perplexity': perplexity,
            'val/loss': loss,
        }, prog_bar=True, sync_dist=True)

        # Ensure that the model has no gradients
        self.model.zero_grad()

        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        return self._step(batch, 'train')

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        with torch.inference_mode(mode=False):
            return self._step(batch, 'val')
