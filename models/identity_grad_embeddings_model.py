import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from config import LLMPagConfig
from data.data_processor import BatchType
from models.base_model import BaseLMModel
from models.common import forward_grad_embeddings, compute_top_k_accuracies
from utils.ngram_processor import NGramProcessor


class IdentityGradEmbeddingsModel(BaseLMModel):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerFast,
            config: LLMPagConfig,
            model_name: str = 'identity',
            num_iterations: int = 1
    ):
        super().__init__(model_name, model, tokenizer, config)
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
        if None in self.mask_values:
            print("Warning: None value found in mask_values, removing it.")
            self.mask_values.remove(None)
        print(f"Test model with mask values: {self.mask_values}")

        # Use default normalization layer
        self.emb_norm = None
        
        # Sign multiplier for gradient logits (typically 1 or -1)
        self.grad_logits_sign = 1
        
        # Initialization methods for inverse validation
        self.inverse_init_methods = [] #['random', 'constant', 'ngram_based']
        self.num_iterations = num_iterations  # Default number of iterations for inverse generation
        self.tokens_per_iteration = 1  # Number of tokens to predict per iteration (default is 1, can be adjusted)
        
        # N-gram processor for inverse validation
        self.ngram_order = 2  # Default to bigram (predict based on n future tokens)
        self.ngram_processor = NGramProcessor(
            ngram_order=self.ngram_order,
            cache_dir=config.model.output_dir / "ngram_cache" if hasattr(config.model, 'output_dir') else "./cache/ngrams",
            mask_values=self.mask_values,
            vocab_size=model.config.vocab_size,
        )
        
        # Initialize structures for different initialization methods
        self.constant_token_id = self.tokenizer.mask_token_id

    def inverse_forward(
        self,
        initial_token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shift_labels: torch.Tensor,
        original_tokens: torch.Tensor = None
    ) -> tuple[torch.Tensor, dict]:
        """
        Core inverse generation engine that performs iterative refinement to predict original tokens
        from initial embeddings using gradient information.
        
        Args:
            initial_token_ids: Starting token IDs of shape (batch_size, sequence_length)
            attention_mask: Attention mask indicating valid positions
            shift_labels: Target sequence for computing CE loss
            original_tokens: Ground-truth tokens for metric computation (optional)
            
        Returns:
            tuple: (final_predicted_tokens, metrics_history)
                - final_predicted_tokens: Final token predictions after all iterations
                - metrics_history: Dictionary containing metrics for each iteration (if original_tokens provided)
        """
        # Clone inputs to avoid in-place modifications
        current_token_ids = initial_token_ids.clone()
        
        # Get original valid tokens for metric computation if provided
        original_valid_tokens = None
        if original_tokens is not None:
            original_valid_tokens = original_tokens[attention_mask == 1]
            if original_valid_tokens.numel() == 0:
                return current_token_ids, {}
        
        # Track metrics across iterations
        metrics_history = {
            'losses': [],
            'accuracies': [],
            'top_k_accuracies': []
        }
        
        for iteration in range(self.num_iterations):
            # Get embeddings for current tokens
            current_x_embed = self.model.get_input_embeddings()(current_token_ids.detach())
            current_x_embed.requires_grad_(True)

            # Forward pass with current embeddings to get CE loss
            outputs_current: CausalLMOutputWithPast = self.model(
                inputs_embeds=current_x_embed,
                attention_mask=attention_mask,
                labels='dummy',
                shift_labels=shift_labels,
                output_hidden_states=False
            )
            loss_ce_current = outputs_current.loss

            if loss_ce_current is None:
                break  # Skip remaining iterations if loss computation fails

            # Get gradients of this CE loss w.r.t. the current embeddings
            grad_wrt_current_x_embed: torch.Tensor | None = None
            try:
                grad_wrt_current_x_embed = torch.autograd.grad(
                    outputs=loss_ce_current,
                    inputs=[current_x_embed],
                    create_graph=False,
                    allow_unused=False
                )[0]
            except RuntimeError:
                break  # Skip remaining iterations if gradient computation fails

            if grad_wrt_current_x_embed is None:
                break

            # Filter gradients and embeddings to valid token positions
            grad_filtered = grad_wrt_current_x_embed[attention_mask == 1]
            current_embed_filtered = current_x_embed[attention_mask == 1]

            if grad_filtered.numel() == 0:
                break

            # Create log info for this iteration
            log_info_iter = {}

            # Use PAG head to predict tokens
            logits_pag_iter = forward_grad_embeddings(
                self.model,
                current_embed_filtered - grad_filtered,
                norm=self.emb_norm,
                log_info=log_info_iter,
                tag=f'inverse_gen_iter_{iteration}'
            )

            # Apply sign to logits
            logits_pag_iter = logits_pag_iter * self.grad_logits_sign

            # Compute metrics if original tokens are provided
            if original_valid_tokens is not None:
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
                    f'inverse_gen_iter_{iteration}'
                )

                # Store metrics for this iteration
                metrics_history['losses'].append(loss_pag_iter.item())
                metrics_history['accuracies'].append(accuracy_iter.item())
                metrics_history['top_k_accuracies'].append(top_k_accuracies_iter)

            # Get the most probable tokens from logits and update current_token_ids
            predicted_token_ids_filtered = torch.argmax(logits_pag_iter, dim=-1)

            # Update the token IDs at valid positions with predicted tokens
            current_token_ids_flat = current_token_ids.view(-1)
            valid_positions = (attention_mask == 1).view(-1)
            current_token_ids_flat[valid_positions] = predicted_token_ids_filtered
            current_token_ids = current_token_ids_flat.view(current_token_ids.shape)

        return current_token_ids, metrics_history

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

        # Calculate gradient-based loss if we're past the warmup period
        if self.current_epoch >= self.warmup_pretrain_epochs:
            # Get the gradients on the first token
            grad_x_embed = torch.autograd.grad(loss_ce, [x_embed], create_graph=create_graph)[0]
            # Take only the valid gradients
            grad_x_embed = grad_x_embed[batch.attention_mask == 1]

            n_first, vocab_size, d = valid_tokens.size(0), self.model.config.vocab_size, x_embed.size(-1)
            assert grad_x_embed.shape == (n_first, d), \
                f'Expected grad_x_embed to be of shape (n\'={n_first}, {d=}), but got {grad_x_embed.shape}'

            # Forward pass to get the logits and probabilities
            logits = forward_grad_embeddings(self.model, x_embed[batch.attention_mask == 1] - grad_x_embed)
            assert logits.shape == (n_first, vocab_size), \
                f'Expected logits to be of shape (n\'={n_first}, {vocab_size=}), but got {logits.shape}'

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

    def inverse_validation(self, batch: BatchType, batch_idx: int, tag: str = 'val') -> None:
        """
        Perform inverse validation using all initialization methods and log results separately.
        
        Args:
            batch (BatchType): The batch of data.
            batch_idx (int): The batch index.
            tag (str): Tag prefix for logging (e.g., 'val', 'test').
        """
        # Test each initialization method
        for method in self.inverse_init_methods:
            # Choose validation approach based on method
            if method == 'ngram_based':
                self._inverse_validation_sequential_with_method(batch, batch_idx, method, tag)
            else:
                self._inverse_validation_parallel_with_method(batch, batch_idx, method, tag)

    def _inverse_validation_parallel_with_method(self, batch: BatchType, batch_idx: int, method: str, tag: str = 'val') -> None:
        """
        Parallel inverse validation - predict all tokens simultaneously using the core inverse_generate engine.
        
        Args:
            batch (BatchType): The batch of data.
            batch_idx (int): The batch index.
            method (str): The initialization method to use.
            tag (str): Tag prefix for logging (e.g., 'val', 'test').
        """
        # Clone data from batch to avoid in-place modifications
        input_ids_cloned = batch.input_ids.clone()
        attention_mask_cloned = batch.attention_mask.clone()
        shift_labels_cloned = batch.shift_labels.clone()

        # Original tokens at valid positions (these will be our target for the PAG head)
        original_valid_tokens = input_ids_cloned[attention_mask_cloned == 1]

        if original_valid_tokens.numel() == 0:
            return

        # Create initial embeddings using the specified initialization method
        initial_token_ids = self._get_initial_tokens(input_ids_cloned.shape, input_ids_cloned.device, method)
        
        # Use the core inverse generation engine
        final_token_ids, metrics_history = self.inverse_forward(
            initial_token_ids=initial_token_ids,
            attention_mask=attention_mask_cloned,
            shift_labels=shift_labels_cloned,
            original_tokens=input_ids_cloned
        )

        # Extract metrics from history for logging
        iteration_losses = metrics_history.get('losses', [])
        iteration_accuracies = metrics_history.get('accuracies', [])
        iteration_top_k_accuracies = metrics_history.get('top_k_accuracies', [])

        # Log iteration-specific metrics (optional, might be too verbose)
        for iteration, (loss, accuracy, top_k_acc) in enumerate(zip(iteration_losses, iteration_accuracies, iteration_top_k_accuracies)):
            if iteration % 2 == 0 or iteration == len(iteration_losses) - 1:  # Log every 2 iteration + last
                self.log_dict({
                    f'{tag}_inverse_{method}/iter_{iteration}_loss': loss,
                    f'{tag}_inverse_{method}/iter_{iteration}_acc': accuracy,
                    **{f'{tag}_inverse_{method}/{k}': v for k, v in top_k_acc.items()}
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

            # Get final top-k accuracies from the last iteration
            final_accuracies = iteration_top_k_accuracies[-1] if iteration_top_k_accuracies else {}

            # Log comprehensive metrics with method-specific names for easy comparison
            self.log_dict({
                # Method-specific metrics
                f'{tag}_inverse_{method}/final_loss': final_loss,
                f'{tag}_inverse_{method}/final_accuracy': final_accuracy,
                f'{tag}_inverse_{method}/initial_loss': initial_loss,
                f'{tag}_inverse_{method}/initial_accuracy': initial_accuracy,
                f'{tag}_inverse_{method}/loss_improvement': loss_improvement,
                f'{tag}_inverse_{method}/accuracy_improvement': accuracy_improvement,
                f'{tag}_inverse_{method}/iterations_completed': len(iteration_losses),
                
                **{f'{tag}_inverse_{method}/{k}': v for k, v in final_accuracies.items()}
            }, prog_bar=False, sync_dist=True)

    def _inverse_validation_sequential_with_method(self, batch: BatchType, batch_idx: int, method: str, tag: str = 'val') -> None:
        """
        Sequential inverse validation - predict tokens one by one from last to first using n-gram statistics.
        This approach is specifically designed for n-gram based initialization.
        
        Args:
            batch (BatchType): The batch of data.
            batch_idx (int): The batch index.
            method (str): The initialization method to use.
            tag (str): Tag prefix for logging (e.g., 'val', 'test').
        """
        # Clone data from batch to avoid in-place modifications
        input_ids_cloned = batch.input_ids.clone()
        attention_mask_cloned = batch.attention_mask.clone()
        shift_labels_cloned = batch.shift_labels.clone()

        # Original tokens at valid positions (these will be our target)
        original_valid_tokens = input_ids_cloned[attention_mask_cloned == 1]

        if original_valid_tokens.numel() == 0:
            return

        # Create initial embeddings using the specified initialization method
        initial_token_ids = self._get_initial_tokens(input_ids_cloned.shape, input_ids_cloned.device, method)
        current_token_ids = initial_token_ids.clone()

        # Get the shape information
        batch_size, seq_len = input_ids_cloned.shape
        device = input_ids_cloned.device

        # Track metrics for n-gram predictions
        total_predictions = 0
        correct_predictions = 0

        # Process each sequence in the batch for n-gram predictions
        for batch_idx_seq in range(batch_size):
            sequence_mask = attention_mask_cloned[batch_idx_seq]
            valid_positions = torch.where(sequence_mask == 1)[0]
            
            if len(valid_positions) <= self.ngram_order:
                continue  # Skip sequences that are too short for n-gram prediction

            # Predict tokens from last to first (excluding the last n tokens which serve as context)
            for pos_idx in range(len(valid_positions) - self.ngram_order - 1, -1, -1):
                current_pos = valid_positions[pos_idx].item()
                
                # Get the context (next N tokens)
                context_start = current_pos + 1
                context_end = min(current_pos + 1 + self.ngram_order, seq_len)
                context_positions = valid_positions[
                    (valid_positions >= context_start) & (valid_positions < context_end)
                ]
                
                if len(context_positions) < self.ngram_order:
                    continue  # Not enough context for n-gram prediction
                
                # Extract context tokens from current prediction
                context_tokens = [current_token_ids[batch_idx_seq, pos].item() for pos in context_positions]
                
                # Predict token using n-gram statistics
                predicted_token = self.ngram_processor.predict_token(context_tokens, device)
                
                # Update the prediction
                current_token_ids[batch_idx_seq, current_pos] = predicted_token
                
                # Check accuracy
                target_token = input_ids_cloned[batch_idx_seq, current_pos].item()
                if predicted_token == target_token:
                    correct_predictions += 1
                total_predictions += 1

        # Now perform iterative refinement using gradients via the core engine
        final_token_ids, metrics_history = self.inverse_forward(
            initial_token_ids=current_token_ids,  # Start with n-gram initialized tokens
            attention_mask=attention_mask_cloned,
            shift_labels=shift_labels_cloned,
            original_tokens=input_ids_cloned
        )

        # Extract metrics from history for logging
        iteration_losses = metrics_history.get('losses', [])
        iteration_accuracies = metrics_history.get('accuracies', [])

        # Log iteration-specific metrics
        for iteration, (loss, accuracy) in enumerate(zip(iteration_losses, iteration_accuracies)):
            if iteration % 2 == 0 or iteration == len(iteration_losses) - 1:
                self.log_dict({
                    f'{tag}_inverse_{method}/iter_{iteration}_loss': loss,
                    f'{tag}_inverse_{method}/iter_{iteration}_acc': accuracy,
                }, prog_bar=False, sync_dist=True)

        # Log n-gram prediction accuracy
        ngram_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Log final metrics
        if iteration_losses:
            final_loss = iteration_losses[-1]
            final_accuracy = iteration_accuracies[-1]
            initial_loss = iteration_losses[0]
            initial_accuracy = iteration_accuracies[0]

            # Compute improvement metrics
            loss_improvement = initial_loss - final_loss
            accuracy_improvement = final_accuracy - initial_accuracy

            # Log comprehensive metrics with method-specific names for easy comparison
            self.log_dict({
                # Method-specific metrics
                f'{tag}_inverse_{method}/ngram_accuracy': ngram_accuracy,
                f'{tag}_inverse_{method}/total_predictions': total_predictions,
                f'{tag}_inverse_{method}/final_loss': final_loss,
                f'{tag}_inverse_{method}/final_accuracy': final_accuracy,
                f'{tag}_inverse_{method}/initial_loss': initial_loss,
                f'{tag}_inverse_{method}/initial_accuracy': initial_accuracy,
                f'{tag}_inverse_{method}/loss_improvement': loss_improvement,
                f'{tag}_inverse_{method}/accuracy_improvement': accuracy_improvement,
                f'{tag}_inverse_{method}/iterations_completed': len(iteration_losses),
            }, prog_bar=False, sync_dist=True)

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        with torch.inference_mode(mode=False):
            # 1. Perform default validation
            original_val_loss = self._step(batch, 'val')  # Logs 'val/...' metrics

            # 2. Perform inverse validation (randomized X validation)
            self.inverse_validation(batch, batch_idx, 'val')

        return original_val_loss  # Return loss from the default validation part

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        with torch.inference_mode(mode=False):
            # 1. Perform default test evaluation
            original_test_loss = self._step(batch, 'test')  # Logs 'test/...' metrics

            # 2. Perform inverse validation (randomized X validation) for testing
            self.inverse_validation(batch, batch_idx, 'test')

        return original_test_loss  # Return loss from the default test part

    def compute_ngram_statistics(self, data_module):
        """
        Compute n-gram statistics from the training dataset using the NGramProcessor.
        This should be called before training starts.
        
        Args:
            data_module: The Lightning data module containing the training data
        """
        print(f"Computing {self.ngram_order}-gram statistics using NGramProcessor...")
        
        # Fit the n-gram processor on the training data
        self.ngram_processor.fit(data_module)
        
        # Print confirmation that n-gram processor is ready
        if 'ngram_based' in self.inverse_init_methods:
            print("N-gram processor ready for sequential prediction in n-gram based initialization")
        
        # Print statistics
        stats = self.ngram_processor.get_statistics()
        print(f"N-gram processor fitted successfully:")
        for key, value in stats.items():
            if key != 'mask_values':  # Skip printing mask values as they can be long
                print(f"  {key}: {value}")

    def _get_initial_tokens(self, input_shape: torch.Size, device: torch.device, method: str) -> torch.Tensor:
        """
        Generate initial tokens based on the specified initialization method.
        
        Args:
            input_shape: Shape of the input tensor (batch_size, sequence_length)
            device: Device to place the tensor on
            method: Initialization method to use
            
        Returns:
            torch.Tensor: Initial token IDs of shape input_shape
        """
            
        if method == 'random':
            return torch.randint(0, self.model.config.vocab_size, input_shape, device=device)
        elif method == 'constant':
            return torch.full(input_shape, self.constant_token_id, device=device, dtype=torch.long)
        elif method == 'ngram_based':
            return torch.randint(0, self.model.config.vocab_size, input_shape, device=device)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def prepare_ngram_statistics(self, data_module):
        """
        Prepare n-gram statistics if using n-gram based initialization.
        This should be called before training starts.
        
        Args:
            data_module: The Lightning data module containing the training data
        """
        if 'ngram_based' in self.inverse_init_methods:
            print("Preparing n-gram statistics for inverse validation...")
            self.compute_ngram_statistics(data_module)
            print("N-gram statistics preparation completed.")
        else:
            print("N-gram based initialization not enabled, skipping n-gram statistics computation.")
