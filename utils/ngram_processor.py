import os
import pickle
import hashlib
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm


class NGramProcessor:
    """
    Efficient N-gram processor with caching capabilities for forward LMs.
    
    This class handles:
    - Computing n-gram statistics from training data (P(token | preceding_context))
    - Caching computed statistics to disk
    - Loading cached statistics from disk
    - Predicting tokens based on n-gram probabilities
    """
    
    def __init__(
        self,
        ngram_order: int = 2,  # N in N-gram (e.g., 2 for bigram, 3 for trigram)
        cache_dir: str = "./cache/ngrams_forward", # Changed cache dir to avoid conflict
        mask_values: Optional[List[int]] = None,
        vocab_size: int = 2048
    ):
        if ngram_order < 2:
            raise ValueError("ngram_order must be at least 2 (for bigrams).")
        
        self.ngram_order = ngram_order # N in N-gram
        self.context_length = ngram_order - 1 # Length of the preceding context
        self.cache_dir = cache_dir
        self.mask_values = set(mask_values) if mask_values else set()
        self.vocab_size = vocab_size
        
        self.ngram_counts: Dict[Tuple[int, ...], Counter] = {}
        # ngram_probs will store P(target_token | context_tuple)
        self.ngram_probs: Dict[Tuple[int, ...], Dict[int, float]] = {} 
        self.context_to_idx: Dict[Tuple[int, ...], int] = {}
        self.idx_to_context: Dict[int, Tuple[int, ...]] = {}
        self.total_ngrams = 0
        self.is_fitted = False
        
        # Tensors will always be dense
        self.context_tensor: Optional[torch.Tensor] = None # Stores unique contexts as rows
        self.prob_tensor: Optional[torch.Tensor] = None # Stores probability distributions
        
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _compute_data_hash(self, data_module: Any) -> str:
        hash_components = [
            str(self.ngram_order), # N value
            str(sorted(list(self.mask_values))),
            str(self.vocab_size),
        ]
        
        if hasattr(data_module, 'data_dir'):
            hash_components.append(str(data_module.data_dir))
        elif hasattr(data_module, 'dataset_name'):
            hash_components.append(str(data_module.dataset_name))
        
        combined_string = "|".join(hash_components)
        return hashlib.md5(combined_string.encode()).hexdigest()
    
    def _get_cache_filepath(self, data_hash: str) -> str:
        # Filename reflects N of N-gram
        filename = f"ngram_stats_{self.ngram_order}gram_dense_{data_hash}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def _save_to_cache(self, cache_filepath: str) -> None:
        cache_data = {
            'ngram_order': self.ngram_order,
            'mask_values': list(self.mask_values),
            'vocab_size': self.vocab_size,
            'ngram_counts': {k: dict(v) for k, v in self.ngram_counts.items()},
            'context_to_idx': self.context_to_idx,
            'idx_to_context': self.idx_to_context,
            'total_ngrams': self.total_ngrams,
            'prob_tensor': self.prob_tensor,
            'context_tensor': self.context_tensor,
        }
        
        try:
            with open(cache_filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved n-gram statistics to cache: {cache_filepath}")
        except Exception as e:
            print(f"Warning: Failed to save n-gram cache: {e}")
    
    def _load_from_cache(self, cache_filepath: str) -> bool:
        try:
            with open(cache_filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            if (cache_data['ngram_order'] != self.ngram_order or 
                set(cache_data['mask_values']) != self.mask_values or
                cache_data.get('vocab_size', self.vocab_size) != self.vocab_size):
                print("Cache validation failed: configuration mismatch")
                return False
            
            self.ngram_counts = {k: Counter(v) for k, v in cache_data['ngram_counts'].items()}
            self.context_to_idx = cache_data.get('context_to_idx', {})
            self.idx_to_context = cache_data.get('idx_to_context', {})
            self.total_ngrams = cache_data['total_ngrams']
            
            self.prob_tensor = cache_data.get('prob_tensor')
            self.context_tensor = cache_data.get('context_tensor')

            if self.prob_tensor is not None and self.prob_tensor.is_sparse:
                print("Warning: Cache contains sparse tensor, but this class version expects dense. Recomputing might be necessary if issues arise.")
                # Attempt to convert, or simply fail loading if strictness is desired.
                # For now, let it load and rebuild_prob_dicts will handle it based on current tensor state.

            self._rebuild_prob_dicts_from_tensor()
            
            self.is_fitted = True
            print(f"Loaded n-gram statistics from cache: {cache_filepath}")
            print(f"  N-gram Order (N): {self.ngram_order}")
            print(f"  Context Length: {self.context_length}")
            print(f"  Total {self.ngram_order}-grams: {self.total_ngrams}")
            print(f"  Unique contexts: {len(self.ngram_counts)}")
            return True
            
        except Exception as e:
            print(f"Warning: Failed to load n-gram cache: {e}")
            return False

    def fit(self, data_module: Any, force_recompute: bool = False) -> None:
        data_hash = self._compute_data_hash(data_module)
        cache_filepath = self._get_cache_filepath(data_hash)
        
        if not force_recompute and os.path.exists(cache_filepath):
            if self._load_from_cache(cache_filepath):
                print(f"Successfully loaded n-gram statistics from cache: {cache_filepath}")
                return
            else:
                print(f"Cache file found at {cache_filepath}, but loading failed or config mismatched. Recomputing.")

        print(f"Computing {self.ngram_order}-gram statistics (context length {self.context_length}) from training data...")
        self._compute_ngram_statistics(data_module)
        self._save_to_cache(cache_filepath)
        self.is_fitted = True

    def _compute_ngram_statistics(self, data_module: Any) -> None:
        self.ngram_counts = {} # Stores context -> Counter(target_token)
        self.total_ngrams = 0
        
        if not hasattr(data_module, 'train_dataset') or data_module.train_dataset is None:
            data_module.setup('fit')
        
        train_dataloader = data_module.train_dataloader()
        
        print("Processing sequences batch by batch for n-gram statistics...")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Processing batches")):
            input_ids = self._extract_input_ids(batch, batch_idx)
            if input_ids is None:
                continue
            
            if isinstance(input_ids, torch.Tensor):
                input_ids_np = input_ids.cpu().numpy()
            elif isinstance(input_ids, np.ndarray):
                input_ids_np = input_ids
            else:
                try:
                    input_ids_np = np.array(input_ids) # Attempt conversion
                except Exception as e:
                    print(f"Warning: Could not convert input_ids to NumPy array for batch {batch_idx}: {e}")
                    continue

            self._process_sequences_batch(input_ids_np)
        
        self._build_tensor_representations()
        
        print(f"Computed n-gram statistics:")
        print(f"  N-gram Order (N): {self.ngram_order}")
        print(f"  Context Length: {self.context_length}")
        print(f"  Total {self.ngram_order}-grams processed: {self.total_ngrams}")
        print(f"  Unique contexts: {len(self.ngram_counts)}")
        if len(self.ngram_counts) > 0:
             print(f"  Average targets per context: {self.total_ngrams / len(self.ngram_counts):.2f}")
        else:
             print(f"  Average targets per context: 0.00")

    def _process_sequences_batch(self, sequences_batch: np.ndarray) -> None:
        for seq_idx in range(sequences_batch.shape[0]):
            sequence = sequences_batch[seq_idx]
            if self._sequence_has_too_many_special_tokens(sequence):
                continue
            self._extract_ngrams_from_sequence(sequence)

    def _sequence_has_too_many_special_tokens(self, sequence: np.ndarray, threshold: float = 0.5) -> bool:
        if not self.mask_values or len(sequence) == 0:
            return False
        num_special = np.sum(np.isin(sequence, list(self.mask_values)))
        return (num_special / len(sequence)) > threshold

    def _extract_ngrams_from_sequence(self, sequence: np.ndarray) -> None:
        seq_len = len(sequence)
        # Need at least context_length + 1 (target) tokens = ngram_order tokens
        if seq_len < self.ngram_order:
            return

        # Iterate from the first possible position where a full n-gram can be formed
        # pos is the index of the target token
        for pos in range(self.context_length, seq_len):
            target_token = sequence[pos].item()

            if target_token in self.mask_values:
                continue

            # Context is tokens PRECEDING the target_token
            context_slice = sequence[pos - self.context_length : pos]
            context_tuple = tuple(context_slice.tolist())

            if self._contains_special_tokens(context_tuple):
                continue
            
            if context_tuple not in self.ngram_counts:
                self.ngram_counts[context_tuple] = Counter()
            self.ngram_counts[context_tuple][target_token] += 1
            self.total_ngrams += 1
            
    def _contains_special_tokens(self, context_tokens: Tuple[int, ...]) -> bool:
        if not self.mask_values:
            return False
        return not self.mask_values.isdisjoint(context_tokens)

    def _build_tensor_representations(self) -> None:
        if not self.ngram_counts:
            self.prob_tensor = torch.empty((0, self.vocab_size), dtype=torch.float32)
            self.context_tensor = torch.empty((0, self.context_length), dtype=torch.long)
            self.context_to_idx = {}
            self.idx_to_context = {}
            return

        self.context_to_idx = {context: idx for idx, context in enumerate(self.ngram_counts.keys())}
        self.idx_to_context = {idx: context for context, idx in self.context_to_idx.items()}
        
        num_contexts = len(self.context_to_idx)
        
        # Build context_tensor
        ordered_contexts = []
        if num_contexts > 0: # Ensure there are contexts to process
            if self.context_length == 0: # Should not happen with ngram_order >= 2
                 ordered_contexts = [[] for _ in range(num_contexts)] # Represent empty contexts if needed
                 self.context_tensor = torch.empty((num_contexts, 0), dtype=torch.long)
            else:
                ordered_contexts = [list(self.idx_to_context[i]) for i in range(num_contexts)]
                self.context_tensor = torch.tensor(ordered_contexts, dtype=torch.long)
        else: # No contexts found
            self.context_tensor = torch.empty((0, self.context_length), dtype=torch.long)


        # Build dense probability tensor
        self.prob_tensor = torch.zeros(num_contexts, self.vocab_size, dtype=torch.float32)
        if num_contexts > 0:
            for context_idx in range(num_contexts):
                context = self.idx_to_context[context_idx]
                target_counts = self.ngram_counts[context]
                total_count = sum(target_counts.values())
                if total_count == 0: continue

                for target, count in target_counts.items():
                    if target < self.vocab_size: # Ensure target is within vocab
                        self.prob_tensor[context_idx, target] = count / total_count
        
        self._rebuild_prob_dicts_from_tensor()

    def _rebuild_prob_dicts_from_tensor(self) -> None:
        self.ngram_probs = {}
        if self.prob_tensor is None or not self.idx_to_context or self.prob_tensor.numel() == 0:
            return

        for context_idx, context_tuple in self.idx_to_context.items():
            probs_for_context = {}
            probs_row = self.prob_tensor[context_idx]
            nz_indices = torch.nonzero(probs_row, as_tuple=True)[0]
            for target_idx in nz_indices:
                probs_for_context[target_idx.item()] = probs_row[target_idx].item()
            
            if probs_for_context:
                self.ngram_probs[context_tuple] = probs_for_context
                
    def _extract_input_ids(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        # (Same as previous version, seems robust enough)
        if isinstance(batch, torch.Tensor):
            return batch
        if hasattr(batch, 'input_ids'):
            return batch.input_ids
        elif isinstance(batch, (list, tuple)) and len(batch) > 0 :
            if isinstance(batch[0], torch.Tensor):
                 return torch.stack(batch) if batch[0].dim() > 0 else torch.tensor(batch)
            elif hasattr(batch[0], 'input_ids'):
                 return torch.stack([item.input_ids for item in batch])
        elif isinstance(batch, dict) and 'input_ids' in batch:
            return batch['input_ids']
        
        print(f"Warning: Could not extract input_ids from batch {batch_idx} of type {type(batch)}")
        return None

    def _predict_token_from_probs_row(self, probs_tensor_row: torch.Tensor, device: torch.device) -> int:
        probs_on_device = probs_tensor_row.to(device)
        if not torch.any(probs_on_device > 0):
            return torch.randint(0, self.vocab_size, (1,), device=device).item()
        
        probs_non_negative = torch.clamp(probs_on_device, min=0)
        if probs_non_negative.sum() == 0:
            return torch.randint(0, self.vocab_size, (1,), device=device).item()

        try:
            sampled_token = torch.multinomial(probs_non_negative, 1).item()
            return sampled_token
        except RuntimeError:
            return torch.randint(0, self.vocab_size, (1,), device=device).item()

    def _predict_token_tensor_internal(self, context_idx: int, device: torch.device) -> int:
        if self.prob_tensor is None or context_idx >= self.prob_tensor.shape[0]:
             return torch.randint(0, self.vocab_size, (1,), device=device).item()
        
        probs_row = self.prob_tensor[context_idx]
        return self._predict_token_from_probs_row(probs_row, device)

    def predict_token(self, context_tokens: List[int], device: torch.device) -> int:
        if not self.is_fitted:
            raise RuntimeError("NGramProcessor must be fitted before making predictions")
        
        # Ensure context_tokens has at least self.context_length tokens
        # If context_length is 0 (for unigram-like behavior, though ngram_order >= 2), handle appropriately.
        # However, self.context_length >= 1 due to ngram_order >= 2.
        
        actual_context_list = context_tokens[-self.context_length:] if self.context_length > 0 else []
        context_tuple = tuple(actual_context_list)
        
        if self.prob_tensor is not None and context_tuple in self.context_to_idx:
            context_idx = self.context_to_idx[context_tuple]
            return self._predict_token_tensor_internal(context_idx, device)
        
        if context_tuple in self.ngram_probs: # Fallback to dict
            prob_dict = self.ngram_probs[context_tuple]
            if prob_dict:
                targets = list(prob_dict.keys())
                probs_values = list(prob_dict.values())
                
                probs_tensor = torch.tensor(probs_values, device=device, dtype=torch.float32)
                if torch.any(probs_tensor > 0):
                    try:
                        sampled_idx = torch.multinomial(probs_tensor, 1).item()
                        return targets[sampled_idx]
                    except RuntimeError:
                        pass
        
        return torch.randint(0, self.vocab_size, (1,), device=device).item()

    def predict_token_batch(self, context_batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        # context_batch shape: (batch_size, self.context_length)
        if not self.is_fitted:
            raise RuntimeError("NGramProcessor must be fitted before making predictions")
        
        batch_size = context_batch.shape[0]
        predictions = torch.randint(0, self.vocab_size, (batch_size,), device=device, dtype=torch.long)

        if self.prob_tensor is None or self.context_tensor is None or self.context_tensor.numel() == 0:
            for i in range(batch_size): # Fallback
                context_list = context_batch[i].cpu().tolist() if self.context_length > 0 else []
                predictions[i] = self.predict_token(context_list, device)
            return predictions

        context_tensor_cpu = self.context_tensor.cpu()
        context_batch_cpu = context_batch.cpu() # Assuming context_batch might be on GPU

        try:
            # Handle empty context_tensor for context_length=0 if it were allowed
            if self.context_length == 0: # e.g. unigram prediction (not standard for this class)
                # If all contexts are effectively the same "empty" context
                if 0 in self.idx_to_context and self.idx_to_context[0] == ():
                    probs_row = self.prob_tensor[0].to(device)
                    # Sample for all items in batch using this single distribution
                    # This requires careful handling if multinomial can take batch of probs
                    # For now, loop for simplicity if this edge case is hit
                    for i in range(batch_size):
                        predictions[i] = self._predict_token_from_probs_row(probs_row, device)
                    return predictions
                # else fall through to random or individual
            
            # Standard case: context_length > 0
            # matches: (batch_size, num_unique_contexts)
            matches = torch.all(context_tensor_cpu.unsqueeze(0) == context_batch_cpu.unsqueeze(1), dim=2)
        except RuntimeError as e:
            print(f"Warning: Failed to compute matches tensor for batch prediction (OOM: {e}). Falling back.")
            for i in range(batch_size):
                context_list = context_batch_cpu[i].tolist() if self.context_length > 0 else []
                predictions[i] = self.predict_token(context_list, device)
            return predictions

        found_in_batch_indices, actual_ctx_tensor_indices = matches.nonzero(as_tuple=True)

        if found_in_batch_indices.numel() > 0:
            probs_for_found_contexts = self.prob_tensor.index_select(0, actual_ctx_tensor_indices).to(device)
            valid_prob_rows_mask = probs_for_found_contexts.sum(dim=1) > 1e-9
            
            if valid_prob_rows_mask.any():
                batch_indices_for_valid_sampling = found_in_batch_indices[valid_prob_rows_mask]
                drawable_probs = probs_for_found_contexts[valid_prob_rows_mask]
                try:
                    sampled_tokens = torch.multinomial(drawable_probs, 1).squeeze(1)
                    predictions[batch_indices_for_valid_sampling] = sampled_tokens
                except RuntimeError:
                    # Fallback for the subset if multinomial fails on the batch
                    for i in range(batch_indices_for_valid_sampling.numel()):
                        original_batch_idx = batch_indices_for_valid_sampling[i].item()
                        # predictions[original_batch_idx] is already random.
                        # Could try individual prediction if desired:
                        # context_idx_for_single = actual_ctx_tensor_indices[valid_prob_rows_mask][i].item()
                        # predictions[original_batch_idx] = self._predict_token_tensor_internal(context_idx_for_single, device)
                        pass # Keep random for now
        return predictions

    def get_context_probability(self, context_tokens: List[int], target_token: int) -> float:
        if not self.is_fitted:
            raise RuntimeError("NGramProcessor must be fitted before getting probabilities")
        
        actual_context_list = context_tokens[-self.context_length:] if self.context_length > 0 else []
        context_tuple = tuple(actual_context_list)
        
        if target_token >= self.vocab_size or target_token < 0: return 0.0

        if self.prob_tensor is not None and context_tuple in self.context_to_idx:
            context_idx = self.context_to_idx[context_tuple]
            if context_idx < self.prob_tensor.shape[0]:
                 return self.prob_tensor[context_idx, target_token].item()
            return 0.0 # Should not happen if context_idx is valid

        if context_tuple in self.ngram_probs: # Fallback to dict
            return self.ngram_probs[context_tuple].get(target_token, 0.0)
        
        return 0.0

    def get_context_probabilities_batch(self, context_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        # context_batch: (batch_size, self.context_length)
        # target_batch: (batch_size,)
        if not self.is_fitted:
            raise RuntimeError("NGramProcessor must be fitted before getting probabilities")

        batch_size = context_batch.shape[0]
        probabilities = torch.zeros(batch_size, dtype=torch.float32, device=context_batch.device)

        if self.prob_tensor is None or self.context_tensor is None or self.context_tensor.numel() == 0:
            for i in range(batch_size): # Fallback
                context_list = context_batch[i].cpu().tolist() if self.context_length > 0 else []
                probabilities[i] = self.get_context_probability(context_list, target_batch[i].cpu().item())
            return probabilities.to(context_batch.device)

        context_tensor_cpu = self.context_tensor.cpu()
        context_batch_cpu = context_batch.cpu()
        
        try:
            if self.context_length == 0: # Unigram-like case (empty context)
                if 0 in self.idx_to_context and self.idx_to_context[0] == (): # Check if the "empty" context exists
                    context_idx = 0
                    probs_row = self.prob_tensor[context_idx] # (vocab_size,)
                    clamped_targets = torch.clamp(target_batch.cpu(), 0, self.vocab_size - 1)
                    valid_target_mask = (target_batch.cpu() < self.vocab_size) & (target_batch.cpu() >=0)
                    
                    gathered_probs = probs_row[clamped_targets]
                    gathered_probs[~valid_target_mask] = 0.0
                    probabilities = gathered_probs
                    return probabilities.to(context_batch.device)
                else: # Empty context not found, all probs 0
                    return probabilities.to(context_batch.device)
            
            matches = torch.all(context_tensor_cpu.unsqueeze(0) == context_batch_cpu.unsqueeze(1), dim=2)
        except RuntimeError as e:
            print(f"Warning: Failed to compute matches tensor for batch probability (OOM: {e}). Falling back.")
            for i in range(batch_size):
                context_list = context_batch_cpu[i].tolist() if self.context_length > 0 else []
                probabilities[i] = self.get_context_probability(context_list, target_batch[i].cpu().item())
            return probabilities.to(context_batch.device)

        found_in_batch_indices, actual_ctx_tensor_indices = matches.nonzero(as_tuple=True)
        relevant_targets = target_batch[found_in_batch_indices].cpu()

        if found_in_batch_indices.numel() > 0:
            probs_for_found_contexts = self.prob_tensor.index_select(0, actual_ctx_tensor_indices)
            clamped_targets = torch.clamp(relevant_targets, 0, self.vocab_size - 1)
            valid_target_mask = (relevant_targets < self.vocab_size) & (relevant_targets >=0)
            
            target_probs = probs_for_found_contexts[torch.arange(relevant_targets.size(0)), clamped_targets]
            target_probs[~valid_target_mask] = 0.0
            probabilities[found_in_batch_indices] = target_probs
        
        return probabilities.to(context_batch.device)

    def get_statistics(self) -> Dict[str, Any]:
        if not self.is_fitted:
            return {"is_fitted": False}
        
        avg_targets = 0.0
        if self.ngram_counts and len(self.ngram_counts) > 0:
            avg_targets = self.total_ngrams / len(self.ngram_counts)

        stats = {
            "is_fitted": True,
            "ngram_order_N": self.ngram_order,
            "context_length": self.context_length,
            "total_ngrams_processed": self.total_ngrams,
            "unique_contexts": len(self.ngram_counts),
            "avg_targets_per_context": avg_targets,
            "vocab_size": self.vocab_size,
            "mask_values_count": len(self.mask_values),
            "tensor_backend": "dense"
        }
        
        if self.prob_tensor is not None:
            stats["tensor_shape"] = list(self.prob_tensor.shape)
            numel = self.prob_tensor.numel()
            if numel > 0 :
                nonzero_count = torch.count_nonzero(self.prob_tensor).item()
                stats["tensor_nnz"] = nonzero_count
                stats["tensor_sparsity"] = 1.0 - (nonzero_count / numel)
            else:
                stats["tensor_nnz"] = 0
                stats["tensor_sparsity"] = 1.0 if numel == 0 else 0.0 # Define for empty tensor
        return stats
    
    def clear_cache(self) -> None:
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("ngram_stats_") and filename.endswith(".pkl"):
                    filepath = os.path.join(self.cache_dir, filename)
                    try:
                        os.remove(filepath)
                        print(f"Removed cache file: {filepath}")
                    except Exception as e:
                        print(f"Warning: Failed to remove cache file {filepath}: {e}")
