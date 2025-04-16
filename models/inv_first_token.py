import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel


class InvFirstTokenModel(BaseLMModel):
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


    # def find_k_nearest_neighbors(self, inv_first_embed: torch.Tensor, k: int = 5) -> torch.Tensor:
    #     """
    #     Find the k-nearest neighbors of the first token in the embedding space.

    #     Args:
    #         inv_first_embed (torch.Tensor): The embeddings of the first token. [batch_size, embed_dim]
    #         k (int): The number of nearest neighbors to find.

    #     Returns:
    #         torch.Tensor: The token indices of the k-nearest neighbors.
    #     """
    #     # Get the embedding matrix of the model
    #     embedding_matrix = self.model.get_input_embeddings().weight # [vocab_size, embed_dim]
    #     # Compute the cosine similarity
    #     cos_sim = F.cosine_similarity(
    #         inv_first_embed.unsqueeze(1),  # [batch_size, 1, embed_dim]
    #         embedding_matrix.unsqueeze(0),  # [1, vocab_size, embed_dim]
    #         dim=2,  # Compute similarity along the embedding dimension
    #     ) # [batch_size, vocab_size]
    #     # Get the top k indices
    #     _, top_k_indices = torch.topk(cos_sim, k, dim=1, largest=True, sorted=False) # [batch_size, k]
    #     return top_k_indices

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
        # Get the logits of the model
        logits = self.model.lm_head(grad_x_embed) # [batch_size, vocab_size]

        # Get the probabilities of the model
        probs = F.softmax(logits, dim=-1) # [batch_size, vocab_size]

        # Get the top k indices
        _, top_k_indices = torch.topk(probs, self.k_samples, dim=-1, largest=True, sorted=False) # [batch_size, k]
        
        return logits, probs, top_k_indices


    def _compute_losses(self, batch: BatchType, prefix_tag: str) -> tuple:
        """
        Compute common losses used in both training and validation steps.
        
        Args:
            batch (BatchType): The batch of data.
            prefix_tag (str): The prefix tag (e.g., 'train' or 'val').
        
        Returns:
            tuple: A tuple containing:
                - loss_ce: Cross-entropy loss
                - loss_grads: Gradient-based loss for first token
                - grad_x_embed: Gradients of embeddings
                - inv_first_label: Original first token labels
        """
        # Get the batch size and sequence length
        n, t = batch.input_ids.shape
        
        if prefix_tag != 'train':
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
        
        # Mask the first token in input_ids
        input_ids[:, 0] = self.tokenizer.pad_token_id

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
            # Get the embedding of the first real token (not [PAD])
            inv_first_label = batch.labels[:, 0].clone()
            # inv_first_label_expanded = inv_first_label.unsqueeze(1).expand(-1, t)
            # inv_first_embed = self.model.get_input_embeddings()(inv_first_label_expanded)

            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=create_graph)[0]

            # Forward pass to get the logits and probabilities
            _, probs, _ = self._forward_grad_embeddings(grad_x_embed[:, 0, :])
            
            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.cross_entropy(
                input=probs,
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
        loss_ce, loss_grads, _, _ = self._compute_losses(batch, prefix_tag)
        
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
            batch (BatchType): The batch of data.
            batch_idx (int): The index of the batch.
        """
        # Set the model to evaluation mode
        self.model.eval()

        with torch.inference_mode(mode=False):
            # Compute losses using common function
            loss_ce, loss_grads, grad_x_embed, inv_first_label = self._compute_losses(batch, prefix_tag)
            
            # Combine losses
            loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

            if grad_x_embed is not None:
                # Get the logits and probabilities
                _, _, nn_indices = self._forward_grad_embeddings(grad_x_embed[:, 0, :])

                # Get the batch size
                n = batch.input_ids.shape[0]

                # Check if the first token is in the k-nearest neighbors
                is_in_k_nearest = torch.zeros((n, self.k_samples), device=inv_first_label.device, dtype=torch.bool) # [batch_size, k]
                for k in range(self.k_samples):
                    is_in_k_nearest[:, k] = (inv_first_label == nn_indices[:, k])
                    
                # Calculate the accuracy on the k-nearest neighbors
                for k in range(self.k_samples):
                    # Log the accuracy at exact k position
                    acc = is_in_k_nearest[:, k].float().mean()
                    self.log(f'val/{k+1}_acc', acc, prog_bar=True)

                    # Log the accuracy for the first k positions
                    acc = is_in_k_nearest[:, :k+1].any(dim=1).float().mean()
                    self.log(f'val/top_{k+1}_acc', acc, prog_bar=True)

            # Calculate perplexity
            perplexity = torch.exp(loss_ce)
            self.log_dict({
                'val/loss_ce': loss_ce,
                'val/loss_first_inv': loss_grads,
                'val/perplexity': perplexity
            }, prog_bar=True)

        # Ensure that the model has no gradients
        self.model.zero_grad()
        
        return loss
