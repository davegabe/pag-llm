import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class GCG:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, num_prefix_tokens: int,
                 num_steps: int, search_width: int, top_k: int):
        """
        Initialize the GCG algorithm.

        Args:
            model: LLM to be attacked
            tokenizer: Tokenizer associated to the given LLM
            num_prefix_tokens: Size of the tokens to use as the attack prefix
            num_steps: Number of iterations to run GCG
            search_width: Top samples to keep at each optimization step
            top_k: Top-k tokens to replace
        """
        self.model = model.eval()
        self.device = self.model.device
        self.tokenizer = tokenizer
        assert isinstance(tokenizer.vocab_size, int) and tokenizer.vocab_size > 0
        self.vocab_size: int = tokenizer.vocab_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_steps = num_steps
        self.search_width = search_width
        self.top_k = top_k

    def run(self, y_message: str, evaluate_every_n_steps: int | None = None,
            stop_after_same_loss_steps: int | None = 10, show_progress: bool = True,
            sample_idx: int | None = None) -> tuple[str, str, int]:
        """
        Run the GCG algorithm on the given model and tokenizer, to generate the y_message text as the suffix.

        Args:
            y_message: The message to be targeted as the desired output.
            evaluate_every_n_steps: Number of steps to run before evaluating the attack.
                                    If None, no evaluation is performed.
            stop_after_same_loss_steps: Number of steps to run with the same GCG loss value before stopping the attack.
                                         If None, no early stopping is performed.
            show_progress: Whether to show a progress bar during the attack.
            sample_idx: Optional index of the sample being attacked, for logging purposes.

        Returns:
            x_attack_str: The final attack prompt.
            y_attack_response: The message returned by the model, after the attack.
            step: The number of steps run before stopping the attack.
        """
        # Tokenize the input message
        y_ids, y_embeds = self._batch_embed_target_message(y_message)
        y_len = y_ids.size(-1)

        # Our input X is a set of num_prefix_tokens random tokens, that will have to be optimized
        x_one_hot = self._generate_random_one_hot()

        previous_losses: list[float] = []
        best_loss_so_far = float('inf')
        best_x_one_hot = None
        best_step = 0

        step: int = 0
        desc = f'Running GCG Attack on sample #{sample_idx}' if sample_idx else 'Running GCG Attack on a sample'
        for step in tqdm(range(self.num_steps), desc=desc, position=1) if show_progress else range(self.num_steps):
            x_one_hot.unsqueeze_(0)
            x_one_hot.requires_grad_(True)
            x_one_hot, loss = self._run_step(x_one_hot, y_ids, y_embeds)

            # Track best loss and corresponding attack
            if loss.item() < best_loss_so_far:
                best_loss_so_far = loss.item()
                best_x_one_hot = x_one_hot.detach().clone()
                best_step = step

            if stop_after_same_loss_steps is not None:
                previous_losses.append(loss.item())

                if len(previous_losses) >= stop_after_same_loss_steps \
                        and max(previous_losses) - min(previous_losses) < 1e-9:
                    # Early stopping
                    break

                previous_losses = previous_losses[-stop_after_same_loss_steps:]

            if evaluate_every_n_steps is not None and step % evaluate_every_n_steps == 0 and step > 0:
                # Log the best attack so far, not the last
                if best_x_one_hot is not None:
                    x_attack_str, y_attack_response = self._evaluate_attack(best_x_one_hot, y_len)
                    log_loss = best_loss_so_far
                else:
                    x_attack_str, y_attack_response = self._evaluate_attack(x_one_hot, y_len)
                    log_loss = loss.item()
                tqdm.write(f'Step: {step}')
                tqdm.write(f'\tBest Attack So Far: "{x_attack_str}"')
                tqdm.write(f'\tLLM Response (best loss: {log_loss:.3f}): "{y_attack_response}"\n')

        # Return the best attack found during all steps
        if best_x_one_hot is not None:
            x_attack_str, y_attack_response = self._evaluate_attack(best_x_one_hot, y_len)
            return x_attack_str, y_attack_response, best_step
        else:
            # Fallback: return last
            x_attack_str, y_attack_response = self._evaluate_attack(x_one_hot, y_len)
            return x_attack_str, y_attack_response, step

    @torch.no_grad()
    def _evaluate_attack(self, x_one_hot: torch.Tensor, y_output_length: int) -> tuple[str, str]:
        """
        Run the attack with the given one-hot encoding of the input prompt.

        Args:
            x_one_hot: One-hot encoding of the input prompt to use for the attack.
            y_output_length: Maximum length of the output to generate.

        Returns:
            x_attack_str: The final attack prompt.
            y_attack_response: The message returned by the model, after the attack.
        """
        x_ids = torch.argmax(x_one_hot, dim=-1)
        x_len = x_ids.size(-1)
        x_attack_str = self.tokenizer.decode(x_ids)

        y_attack_response_ids = self.model.generate(
            inputs=x_ids.unsqueeze(0),
            max_new_tokens=y_output_length,
        ).squeeze(0)[x_len:] # type: ignore
        y_attack_response = self.tokenizer.decode(y_attack_response_ids)

        return x_attack_str, y_attack_response

    def _compute_gcg_loss(self, attack_one_hot: torch.Tensor, target_ids: torch.Tensor,
                          target_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute the GCG loss on the attack_one_hot input.

        The input can be either:
        - batched, with shape (batch_size, prefix_len, vocab_size)
        - or a single item, with shape (prefix_len, vocab_size)

        Args:
            attack_one_hot: Input one-hot encoding to compute the loss against.
            target_ids: Desired LLM output token IDs.
            target_embeds: Desired LLM output token embeddings.

        Returns:
            torch.Tensor: GCG Loss per each sample in the attack_one_hot batch.
        """
        assert attack_one_hot.ndim in (2, 3)
        input_is_batched = attack_one_hot.ndim == 3
        if not input_is_batched:
            attack_one_hot = attack_one_hot.unsqueeze(0)
        batch_size = attack_one_hot.size(0)

        target_ids, target_embeds = target_ids[:batch_size], target_embeds[:batch_size]

        attack_embeds = self._embed_one_hot(attack_one_hot)

        full_text_embeds = torch.cat([
            attack_embeds,
            target_embeds,
        ], dim=1)

        # Compute the CE-loss, using teacher forcing.
        # The use of teacher forcing SHOULD be okay, since it is used in nanoGCG:
        # https://github.com/GraySwanAI/nanoGCG/blob/7d45952b0e75131025a44985a75306593e1bd69f/nanogcg/gcg.py#L478
        # From the logits, take only the ones that correspond to the attack target IDs
        output_logits = self.model(inputs_embeds=full_text_embeds).logits[:, self.num_prefix_tokens - 1:-1]

        per_sample_loss = F.cross_entropy(
            input=output_logits.reshape(-1, self.vocab_size),
            target=target_ids.view(-1),
            reduction='none',
        ).view(batch_size, -1).mean(dim=1)

        return per_sample_loss

    def _compute_top_k_substitutions(self, attack_one_hot: torch.Tensor, target_ids: torch.Tensor,
                                     target_embeds: torch.Tensor) -> torch.Tensor:
        """
        First step of GCG: compute the top-k substitutions for each token in the attack input IDs.

        Args:
            attack_one_hot: Input IDs to be used for the attack, as one-hot vector.
            target_ids: Desired output token IDs, if the attack is successful.
            target_embeds: Desired output token embeddings, if the attack is successful.

        Returns:
            torch.Tensor: The top-k replacements for each token in the attack input IDs.
        """
        assert attack_one_hot.requires_grad, f'Expected to compute gradients on the attack IDs'
        assert attack_one_hot.ndim == 2 or attack_one_hot.size(0) == 1, \
            f'Expected attack_one_hot to be of shape (1, {self.num_prefix_tokens}, {self.vocab_size}), but got {attack_one_hot.shape}'

        loss = self._compute_gcg_loss(attack_one_hot, target_ids, target_embeds)[0]

        attack_one_hot_grads = torch.autograd.grad(loss, attack_one_hot)[0][0]
        attack_one_hot_grads /= attack_one_hot_grads.norm(dim=-1, keepdim=True)

        # Now, for each token in the prefix, we need to find the top-k replacements with the lowest gradient
        best_replacements = torch.topk(attack_one_hot_grads, self.top_k, dim=-1, largest=False).indices

        return best_replacements

    @torch.no_grad()
    def _compute_random_replacements(self, attack_one_hot: torch.Tensor,
                                     top_k_substitutions: torch.Tensor) -> torch.Tensor:
        """
        Compute the random replacements for the attack input IDs.

        Args:
            attack_one_hot: Input IDs to be used for the attack, as one-hot vector.
            top_k_substitutions: Top-k substitutions for each token in the attack input IDs.

        Returns:
            torch.Tensor: The candidates prefixes with random replacements for each token.
        """
        # Create the batch of candidates
        candidates = attack_one_hot.repeat(self.search_width, 1, 1)

        # For each candidate, pick a random token index to replace,
        # and then for every token, choose a random replacement from its top-k substitutions
        token_indexes = torch.randint(0, self.num_prefix_tokens, (self.search_width,), device=self.device)

        top_k_token_replacements = top_k_substitutions[token_indexes]
        top_k_random_replacement_indexes = torch.randint(0, self.top_k, (self.search_width,), device=self.device)

        # Now, take replacement tokens from the top-k substitutions.
        # Replacement token for candidate i-th is the top-k substitution for token i, at index top_k_random_replacement[i]
        search_width_range = torch.arange(self.search_width, device=self.device)
        top_k_random_replacement = top_k_token_replacements[search_width_range, top_k_random_replacement_indexes]
        one_hot_random_replacement = F.one_hot(top_k_random_replacement, self.vocab_size).to(dtype=torch.float)
        candidates[search_width_range, token_indexes] = one_hot_random_replacement

        return candidates

    def _run_step(self, attack_one_hot: torch.Tensor, target_ids: torch.Tensor,
                  target_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a GCG attack step, optimizing the attack input IDs.

        Args:
            attack_one_hot: Input IDs to be used for the attack, as one-hot vector.
            target_ids: Desired output token IDs, if the attack is successful.
            target_embeds: Desired output token embeddings, if the attack is successful.

        Returns:
            torch.Tensor: The optimized attack input IDs, as one-hot vector.
            torch.Tensor: The loss of the optimized attack input IDs.
        """
        assert attack_one_hot.ndim == 3 and attack_one_hot.size(0) == 1, \
            f'Expected attack_one_hot to be of shape (1, {self.num_prefix_tokens}, {self.vocab_size}), but got {attack_one_hot.shape}'

        top_k_substitutions = self._compute_top_k_substitutions(attack_one_hot, target_ids, target_embeds)
        assert top_k_substitutions.shape == (self.num_prefix_tokens, self.top_k), \
            f'Expected top_k_substitutions to be of shape ({self.num_prefix_tokens}, {self.top_k}), but got {top_k_substitutions.shape}'

        proposed_prefixes = self._compute_random_replacements(attack_one_hot, top_k_substitutions)

        with torch.no_grad():
            prefixes_loss = self._compute_gcg_loss(proposed_prefixes, target_ids, target_embeds)
            best_prefix_index = torch.argmin(prefixes_loss)

        best_prefix_loss = prefixes_loss[best_prefix_index]
        best_prefix = proposed_prefixes[best_prefix_index]
        return best_prefix, best_prefix_loss

    @torch.no_grad()
    def _batch_embed_target_message(self, target_message: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Embed the target message, using the model's embeddings matrix.
        The target message will be returned as a batch with batch_size = the top-k attack samples.

        Args:
            target_message (str): The message to be targeted.

        Returns:
            torch.Tensor: Tokenized target message, as a batch of input IDs of size top-k.
            torch.Tensor: Embedded target message, as a batch of input embeds of size top-k.
        """
        # Tokenize the input message
        target_ids = self.tokenizer(target_message, return_tensors="pt").input_ids.to(self.device)
        batch_target_ids = target_ids.repeat(self.search_width, 1)

        target_embeds = self.model.get_input_embeddings()(target_ids)
        batch_target_embeds = target_embeds.repeat(self.search_width, 1, 1)

        return batch_target_ids, batch_target_embeds

    def _embed_one_hot(self, one_hot: torch.Tensor) -> torch.Tensor:
        """
        Embed an input tensor, given as the one-hot encoding.

        Args:
            one_hot (torch.Tensor): Input IDs, as one-hot encoding

        Returns:
            torch.Tensor: Embedded input IDs
        """
        return one_hot @ self.model.get_input_embeddings().weight.data # type: ignore

    @torch.no_grad()
    def _generate_random_one_hot(self) -> torch.Tensor:
        """
        Generate random one-hot encodings for random tokens in the vocabulary.

        Returns:
            torch.Tensor: Single one-hot encoding of a sequence with randomly selected tokens in the vocabulary space.
        """
        input_ids = torch.randint(0,
                                  self.vocab_size,
                                  (self.num_prefix_tokens,),
                                  dtype=torch.long,
                                  device=self.device)
        return F.one_hot(input_ids, self.vocab_size).to(dtype=torch.float)
