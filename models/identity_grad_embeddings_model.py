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
            grad_logits_sign: int = 1,
            model_name: str = 'identity',
    ):
        super().__init__(model_name, model, tokenizer, config)
        self.lambda_loss_ce = config.training.lambda_loss_ce
        self.lambda_loss_pag = config.training.lambda_loss_pag
        self.lambda_loss_entropy = 1
        self.warmup_pretrain_epochs = config.training.warmup_pretrain_epochs
        self.grad_logits_sign = grad_logits_sign
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
        if None in self.mask_values:
            self.mask_values.remove(None)
        print(f"Test model with mask values: {self.mask_values}")

        # Use default normalization layer
        self.emb_norm = None

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

        # With probability P, create randomized embeddings (with random tokens)
        if torch.rand(1).item() < 2.0 and tag == 'train' and self.current_epoch >= self.warmup_pretrain_epochs:
            # Create random token IDs
            vocab_size = self.model.config.vocab_size
            random_token_ids = torch.randint(
                0, vocab_size, input_ids.shape, device=input_ids.device
            )
            # random_token_ids = torch.ones_like(input_ids) # DEBUG: Use ones to avoid random tokens, for debugging purposes
            # Get the embeddings for the random tokens
            x_embed = self.model.get_input_embeddings()(random_token_ids)
            x_embed.requires_grad_(True)

            randomized_embeddings = True

        # Otherwise we use the original embeddings
        else:
            # Get the embeddings of X
            x_embed = self.model.get_input_embeddings()(input_ids)
            if create_graph:
                x_embed.requires_grad_(True)

            # Add random noise to the embeddings, use a small norm based on the strength of the embeddings
            if tag == 'train':
                norms = x_embed.norm(dim=-1, keepdim=True)  # [batch, seq, 1]
                noise = torch.randn_like(x_embed) * norms * 0.03  # Scale noise by 3% of the embedding norm
                x_embed = x_embed + noise

            randomized_embeddings = False

        # Log whether we used randomized embeddings
        self.log(f'{tag}/randomized_embeddings', randomized_embeddings, prog_bar=False, sync_dist=True)

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
            x_embed = x_embed[batch.attention_mask == 1]

            n_first, vocab_size, d = valid_tokens.size(0), self.model.config.vocab_size, x_embed.size(-1)
            assert grad_x_embed.shape == (n_first, d), \
                f'Expected grad_x_embed to be of shape (n\'={n_first}, {d=}), but got {grad_x_embed.shape}'

            # Normalize the gradients based on the norm of the gradient
            grad_x_embed = grad_x_embed / (grad_x_embed.norm(dim=-1, keepdim=True) + 1e-8)

            # Dictionary to store gradient information for logging
            log_info = {}
            # Forward pass to get the logits and probabilities
            logits = forward_grad_embeddings(
                self.model,
                grad_x_embed + x_embed,
                norm=self.emb_norm,
                log_info=log_info,
                tag=tag
            )

            # Log the log_info dictionary
            self.log_dict(log_info, sync_dist=True)

            # Apply the sign to the logits
            logits = logits * self.grad_logits_sign

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

        # Combine losses using lambda parameters
        if randomized_embeddings:
            # If we used randomized embeddings, we want to:
            # 1. Maximize the entropy of the model's predictions (lower confidence).
            # 2. Minimize the gradient-based PAG loss (loss_grads).

            # loss_ce (outputs.loss) is CE(model(random_embed), original_shift_labels).
            # It's used for calculating grad_x_embed for loss_grads.

            # Calculate mean entropy of the model's output distribution
            pred_logits = outputs.logits  # Logits corresponding to shift_labels
            
            probs = F.softmax(pred_logits, dim=-1)
            log_probs = F.log_softmax(pred_logits, dim=-1)
            entropy_full = -torch.sum(probs * log_probs, dim=-1) # Shape (batch, seq_len_for_logits)

            pad_token_id = self.tokenizer.pad_token_id
            current_shift_labels = batch.shift_labels

            if pad_token_id is None:
                 if hasattr(batch, 'shift_labels') and batch.shift_labels is not None:
                    mask = torch.ones_like(batch.shift_labels, dtype=entropy_full.dtype, device=entropy_full.device)
                 else: 
                    mask = torch.ones_like(entropy_full, dtype=entropy_full.dtype, device=entropy_full.device)
            else:
                if not hasattr(batch, 'shift_labels') or batch.shift_labels is None:
                    raise ValueError("batch.shift_labels is required for entropy masking when pad_token_id is present.")
                
                # Adjust shapes if there's a mismatch typical of Causal LMs (logits seq_len vs labels seq_len)
                if entropy_full.shape[1] != current_shift_labels.shape[1]:
                    if entropy_full.shape[1] == current_shift_labels.shape[1] - 1 and current_shift_labels.shape[1] > 1:
                        current_shift_labels = current_shift_labels[:, :entropy_full.shape[1]]
                    elif current_shift_labels.shape[1] == entropy_full.shape[1] - 1 and entropy_full.shape[1] > 1:
                        entropy_full = entropy_full[:, :current_shift_labels.shape[1]]

                mask = (current_shift_labels != pad_token_id).type_as(entropy_full)
                if entropy_full.shape[1] != mask.shape[1]:
                     pass 

            masked_entropy_sum = torch.sum(entropy_full * mask)
            num_active_elements = torch.sum(mask)
            
            if num_active_elements > 0:
                mean_entropy = masked_entropy_sum / num_active_elements
            else:
                mean_entropy = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)

            log_vocab_size = torch.log(torch.tensor(self.model.config.vocab_size, dtype=mean_entropy.dtype, device=mean_entropy.device))
            loss_entropy_objective = log_vocab_size - mean_entropy
            
            self.log(f'{tag}/mean_entropy_random', mean_entropy, prog_bar=False, sync_dist=True)
            self.log(f'{tag}/loss_entropy_objective_random', loss_entropy_objective, prog_bar=False, sync_dist=True)
            
            loss = self.lambda_loss_entropy * loss_entropy_objective + self.lambda_loss_pag * loss_grads

        else: # not randomized_embeddings
            # If we used the original embeddings, we want to minimize both CE and PAG losses.
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
        # Ensure gradients can be computed if needed, but don't track for main model parameters here
        # mode=False allows requires_grad=True on intermediate tensors to work for autograd.
        with torch.inference_mode(mode=False):
            # 1. Perform default validation
            original_val_loss = self._step(batch, 'val') # Logs 'val/...' metrics

            # --- Start of Randomized X validation ---
            # This part needs gradients for intermediate computations.

            # Clone data from batch to avoid in-place modifications
            input_ids_cloned = batch.input_ids.clone()
            attention_mask_cloned = batch.attention_mask.clone()
            shift_labels_cloned = batch.shift_labels.clone() # These are the targets for the CE loss

            # Original tokens at valid positions (these will be our target for the PAG head)
            original_valid_tokens = input_ids_cloned[attention_mask_cloned == 1]

            if original_valid_tokens.numel() == 0:
                # print(f"Skipping randomized X validation for batch {batch_idx} due to no valid tokens.")
                return original_val_loss

            # Create randomized input embeddings
            vocab_size = self.model.config.vocab_size
            random_token_ids = torch.randint(
                0, vocab_size, input_ids_cloned.shape, device=input_ids_cloned.device
            )
            # random_token_ids = torch.ones_like(input_ids_cloned)  # DEBUG: Use ones to avoid random tokens, for debugging purposes
            # Detach random_token_ids to ensure they don't carry unexpected grad history
            random_x_embed = self.model.get_input_embeddings()(random_token_ids.detach())
            random_x_embed.requires_grad_(True)

            # Forward pass with randomized embeddings to get CE loss
            # This CE loss is for predicting `shift_labels_cloned` from `random_x_embed`
            outputs_random: CausalLMOutputWithPast = self.model(
                inputs_embeds=random_x_embed,
                attention_mask=attention_mask_cloned,
                labels='dummy', # Model should use shift_labels internally
                shift_labels=shift_labels_cloned,
                output_hidden_states=False
            )
            loss_ce_random = outputs_random.loss

            if loss_ce_random is None:
                # print(f"Warning: loss_ce_random is None in randomized X validation for batch {batch_idx}. Skipping.")
                return original_val_loss

            # Get gradients of this CE loss w.r.t. the randomized embeddings
            try:
                grad_wrt_random_x_embed_tuple = torch.autograd.grad(
                    outputs=loss_ce_random,
                    inputs=[random_x_embed],
                    create_graph=False, # No second-order gradients needed for validation
                    allow_unused=False
                )
                grad_wrt_random_x_embed = grad_wrt_random_x_embed_tuple[0]
            except RuntimeError as e:
                # print(f"RuntimeError during grad computation in randomized X validation for batch {batch_idx}: {e}. Skipping.")
                return original_val_loss

            if grad_wrt_random_x_embed is None:
                # print(f"Warning: grad_wrt_random_x_embed is None in randomized X validation for batch {batch_idx}. Skipping.")
                return original_val_loss

            # Filter gradients and the random embeddings to valid token positions
            grad_filtered = grad_wrt_random_x_embed[attention_mask_cloned == 1]
            random_embed_filtered = random_x_embed[attention_mask_cloned == 1]

            if grad_filtered.numel() == 0: # Should be caught by original_valid_tokens.numel() check earlier
                return original_val_loss

            # Normalize the gradients
            norm_val = grad_filtered.norm(dim=-1, keepdim=True)
            grad_normalized = grad_filtered / (norm_val + 1e-8) # Epsilon for stability

            # Iterative refinement process
            num_iterations = 10
            current_token_ids = random_token_ids.clone()  # Start with random tokens

            # Track metrics across iterations
            iteration_losses = []
            iteration_accuracies = []

            for iteration in range(num_iterations):
                # Get embeddings for current tokens
                current_x_embed = self.model.get_input_embeddings()(current_token_ids.detach())
                current_x_embed.requires_grad_(True)

                # Forward pass with current embeddings to get CE loss
                outputs_current: CausalLMOutputWithPast = self.model(
                    inputs_embeds=current_x_embed,
                    attention_mask=attention_mask_cloned,
                    labels='dummy',
                    shift_labels=shift_labels_cloned,
                    output_hidden_states=False
                )
                loss_ce_current = outputs_current.loss

                if loss_ce_current is None:
                    break  # Skip remaining iterations if loss computation fails

                # Get gradients of this CE loss w.r.t. the current embeddings
                try:
                    grad_wrt_current_x_embed_tuple = torch.autograd.grad(
                        outputs=loss_ce_current,
                        inputs=[current_x_embed],
                        create_graph=False,
                        allow_unused=False
                    )
                    grad_wrt_current_x_embed = grad_wrt_current_x_embed_tuple[0]
                except RuntimeError as e:
                    break  # Skip remaining iterations if gradient computation fails

                if grad_wrt_current_x_embed is None:
                    break

                # Filter gradients and embeddings to valid token positions
                grad_filtered = grad_wrt_current_x_embed[attention_mask_cloned == 1]
                current_embed_filtered = current_x_embed[attention_mask_cloned == 1]

                if grad_filtered.numel() == 0:
                    break

                # Normalize the gradients
                norm_val = grad_filtered.norm(dim=-1, keepdim=True)
                grad_normalized = grad_filtered / (norm_val + 1e-8)

                # Input for the PAG head: normalized gradient + corresponding current embedding
                pag_head_input = grad_normalized + current_embed_filtered

                log_info_iter = {}

                # Use PAG head to predict tokens
                logits_pag_iter = forward_grad_embeddings(
                    self.model,
                    pag_head_input,
                    norm=self.emb_norm,
                    log_info=log_info_iter,
                    tag=f'val_random_x_pag_iter_{iteration}'
                )

                # Apply sign to logits
                logits_pag_iter = logits_pag_iter * self.grad_logits_sign

                # Compute loss and accuracy for this iteration
                loss_pag_iter = F.cross_entropy(
                    input=logits_pag_iter,
                    target=original_valid_tokens,
                    reduction='mean'
                )

                # Compute top-1 accuracy for this iteration
                predicted_tokens = torch.argmax(logits_pag_iter, dim=-1)
                accuracy_iter = (predicted_tokens == original_valid_tokens).float().mean()

                # Compute top-k accuracies for k=[1,2,3] for this iteration
                top_k_accuracies_iter = compute_top_k_accuracies(
                    original_valid_tokens,
                    logits_pag_iter,
                    3,  # Compute for k=1,2,3
                    f'val_random_x_pag_iter_{iteration}'
                )

                iteration_losses.append(loss_pag_iter.item())
                iteration_accuracies.append(accuracy_iter.item())

                # Get the most probable tokens from logits and update current_token_ids
                predicted_token_ids_filtered = torch.argmax(logits_pag_iter, dim=-1)

                # Update the token IDs at valid positions with predicted tokens
                current_token_ids_flat = current_token_ids.view(-1)
                valid_positions = (attention_mask_cloned == 1).view(-1)
                current_token_ids_flat[valid_positions] = predicted_token_ids_filtered
                current_token_ids = current_token_ids_flat.view(current_token_ids.shape)

                # Log iteration-specific metrics (optional, might be too verbose)
                if iteration % 2 == 0 or iteration == num_iterations - 1:  # Log every 2 iteration + last
                    self.log_dict({
                        f'val_random_x_pag/iter_{iteration}_loss': loss_pag_iter,
                        f'val_random_x_pag/iter_{iteration}_acc': accuracy_iter,
                        **top_k_accuracies_iter  # Include top-k accuracies for this iteration
                    }, prog_bar=False, sync_dist=True)

            # Log final metrics and iteration statistics
            if iteration_losses:
                final_loss = iteration_losses[-1]
                final_accuracy = iteration_accuracies[-1]
                initial_loss = iteration_losses[0]
                initial_accuracy = iteration_accuracies[0]

                # Compute improvement metrics
                loss_improvement = initial_loss - final_loss
                accuracy_improvement = final_accuracy - initial_accuracy

                # Compute final top-k accuracies using the last iteration's logits
                final_accuracies = compute_top_k_accuracies(
                    original_valid_tokens,
                    logits_pag_iter,  # Last iteration's logits
                    3,  # Compute for k=1,2,3
                    'val_random_x_pag_final'
                )

                # Log comprehensive metrics
                self.log_dict({
                    f'val_random_x_pag/final_loss': final_loss,
                    f'val_random_x_pag/final_accuracy': final_accuracy,
                    f'val_random_x_pag/initial_loss': initial_loss,
                    f'val_random_x_pag/initial_accuracy': initial_accuracy,
                    f'val_random_x_pag/loss_improvement': loss_improvement,
                    f'val_random_x_pag/accuracy_improvement': accuracy_improvement,
                    f'val_random_x_pag/iterations_completed': len(iteration_losses),
                    **final_accuracies
                }, prog_bar=False, sync_dist=True)
            # --- End of Randomized X validation ---

        return original_val_loss # Return loss from the default validation part