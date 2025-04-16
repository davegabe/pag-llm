import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel


class InvFirstTokenModel(BaseLMModel):
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
            config: LLMPagConfig
    ):
        super().__init__(model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.k_nearest_neighbors = 5

    def find_k_nearest_neighbors(self, inv_first_embed: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Find the k-nearest neighbors of the first token in the embedding space.

        Args:
            inv_first_embed (torch.Tensor): The embeddings of the first token. [batch_size, embed_dim]
            k (int): The number of nearest neighbors to find.

        Returns:
            torch.Tensor: The token indices of the k-nearest neighbors.
        """
        # Get the embedding matrix of the model
        embedding_matrix = self.model.get_input_embeddings().weight # [vocab_size, embed_dim]
        # Compute the cosine similarity
        cos_sim = F.cosine_similarity(
            inv_first_embed.unsqueeze(1),  # [batch_size, 1, embed_dim]
            embedding_matrix.unsqueeze(0),  # [1, vocab_size, embed_dim]
            dim=2,  # Compute similarity along the embedding dimension
        ) # [batch_size, vocab_size]
        # Get the top k indices
        _, top_k_indices = torch.topk(cos_sim, k, dim=1, largest=True, sorted=False) # [batch_size, k]
        return top_k_indices


    def training_step(self, batch: BatchType, batch_idx: int, prefix_tag: str = 'train') -> torch.Tensor:
        n, t = batch.input_ids.shape

        # Mask the first token in input_ids and labels
        batch.input_ids[:, 0] = self.tokenizer.pad_token_id

        # Get the embeddings of X
        x_embed = self.model.get_input_embeddings()(batch.input_ids)

        # Forward pass, with standard input X
        outputs: CausalLMOutputWithPast = self.model(
            inputs_embeds=x_embed,
            attention_mask=batch.attention_mask,
            labels='dummy',
            shift_labels=batch.shift_labels,
            output_hidden_states=False
        )
        loss_ce = outputs.loss

        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Get the embedding of the first real token (not [PAD])
            inv_first_label = batch.labels[:, 0]
            inv_first_label = inv_first_label.unsqueeze(1).expand(-1, t)
            inv_first_embed = self.model.get_input_embeddings()(inv_first_label)

            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=True)[0]
            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.mse_loss(
                input=grad_x_embed[:, 0, :],
                target=inv_first_embed[:, 0, :]
            )
        else:
            loss_grads = torch.zeros_like(loss_ce)

        loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

        self.log_dict({
            f'{prefix_tag}/loss_ce': loss_ce,
            f'{prefix_tag}/loss_first_inv': loss_grads,
            f'{prefix_tag}/loss': loss,
        }, prog_bar=True)
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int):
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
            input_ids = batch.input_ids.clone()
            attention_mask = batch.attention_mask.clone()
            shift_labels = batch.shift_labels.clone()

            # Get the batch size and sequence length
            n, t = input_ids.shape

            # Mask the first token in input_ids and labels
            input_ids[:, 0] = self.tokenizer.pad_token_id

            # Get the embeddings of X
            x_embed = self.model.get_input_embeddings()(input_ids)
            x_embed.requires_grad_(True)

            # Forward pass, with standard input X
            outputs: CausalLMOutputWithPast = self.model(
                inputs_embeds=x_embed,
                attention_mask=attention_mask,
                labels='dummy',
                shift_labels=shift_labels,
                output_hidden_states=False
            )
            loss_ce = outputs.loss
            
            # Get the embedding of the first real token (not [PAD])
            inv_first_label = batch.labels[:, 0].clone()
            inv_first_label = inv_first_label.unsqueeze(1).expand(-1, t)
            inv_first_embed = self.model.get_input_embeddings()(inv_first_label)

            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=False)[0]
            # We want that gradients on the first token will reconstruct the original token
            loss_grads = F.mse_loss(
                input=grad_x_embed[:, 0, :],
                target=inv_first_embed[:, 0, :]
            )

            loss = self.lambda_loss_ce * loss_ce + self.lambda_loss_pag * loss_grads

            # Find the token indices of the k-nearest neighbors embedding to the first token
            nn_indices = self.find_k_nearest_neighbors(grad_x_embed[:, 0, :], self.k_nearest_neighbors) # [batch_size, k]

            # Get the labels of the first token (token indices)
            inv_first_label = batch.labels[:, 0] # [batch_size]

            # Check if the first token is in the k-nearest neighbors
            is_in_k_nearest = torch.zeros((n, self.k_nearest_neighbors), device=inv_first_label.device, dtype=torch.bool) # [batch_size, k]
            for k in range(self.k_nearest_neighbors):
                is_in_k_nearest[:, k] = (inv_first_label == nn_indices[:, k])
                
            # Calculate the accuracy on the k-nearest neighbors
            for k in range(self.k_nearest_neighbors):
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

        # Ensure that the model has no gradients (this is not necessary, but just to be sure)
        self.model.zero_grad()
        return loss
