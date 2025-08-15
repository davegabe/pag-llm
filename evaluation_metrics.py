"""
Evaluation metrics for backward inference models.

This module implements comprehensive evaluation metrics for assessing the quality
of generated prefixes in backward language modeling tasks.
"""
from typing import Dict, List, Tuple, Any

import evaluate
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss

from models.base_model import BaseLMModel


class BackwardInferenceEvaluator:
    """
    Comprehensive evaluator for backward inference models.
    
    Provides token-level overlap metrics and forward coherence metrics
    to assess the quality of generated prefixes.
    """

    def __init__(self, forward_model: BaseLMModel, external_llm: str, semantic_model: SentenceTransformer):
        """
        Initialize the evaluator.
        
        Args:
            forward_model: A forward language model to compute coherence metrics
            external_llm: Identifier for an external LLM to compute perplexity
            semantic_model: Pre-trained model for semantic similarity evaluation
        """
        self.forward_model = forward_model
        self.forward_model.eval()
        self.external_llm = external_llm
        self.perplexity_evaluator = evaluate.load("perplexity", module_type="metric")
        self.semantic_model = semantic_model
    
    def compute_token_overlap_metrics(self, 
                                    reference_tokens: torch.Tensor,
                                    hypothesis_tokens: torch.Tensor,
                                    reference_attention_mask: torch.Tensor = None,
                                    hypothesis_attention_mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute token-level precision, recall, and F1 scores.
        
        Args:
            reference_tokens: Ground truth prefix tokens [seq_len]
            hypothesis_tokens: Generated prefix tokens [seq_len] 
            reference_attention_mask: Mask for valid reference tokens [seq_len]
            hypothesis_attention_mask: Mask for valid hypothesis tokens [seq_len]
            
        Returns:
            Dictionary containing precision, recall, f1, and overlap statistics
        """
        # Convert to CPU for easier processing
        ref_tokens = reference_tokens.cpu().numpy()
        hyp_tokens = hypothesis_tokens.cpu().numpy()
        
        # Apply attention masks if provided
        if reference_attention_mask is not None:
            ref_mask = reference_attention_mask.cpu().numpy().astype(bool)
            ref_tokens = ref_tokens[ref_mask]
        
        if hypothesis_attention_mask is not None:
            hyp_mask = hypothesis_attention_mask.cpu().numpy().astype(bool)
            hyp_tokens = hyp_tokens[hyp_mask]
        
        # Convert to sets for overlap calculation
        ref_set = set(ref_tokens.tolist())
        hyp_set = set(hyp_tokens.tolist())
        
        # Calculate overlap
        intersection = ref_set.intersection(hyp_set)
        
        # Calculate metrics
        precision = len(intersection) / len(hyp_set) if len(hyp_set) > 0 else 0.0
        recall = len(intersection) / len(ref_set) if len(ref_set) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'token_precision': precision,
            'token_recall': recall, 
            'token_f1': f1
        }
    
    def compute_positional_accuracy(self,
                                  reference_tokens: torch.Tensor,
                                  hypothesis_tokens: torch.Tensor,
                                  reference_attention_mask: torch.Tensor = None,
                                  hypothesis_attention_mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute position-aware accuracy metrics.
        
        Args:
            reference_tokens: Ground truth prefix tokens [seq_len]
            hypothesis_tokens: Generated prefix tokens [seq_len]
            reference_attention_mask: Mask for valid reference tokens [seq_len]  
            hypothesis_attention_mask: Mask for valid hypothesis tokens [seq_len]
            
        Returns:
            Dictionary containing positional accuracy metrics
        """
        ref_tokens = reference_tokens.cpu()
        hyp_tokens = hypothesis_tokens.cpu()
        
        # Determine the comparison length
        min_len = min(len(ref_tokens), len(hyp_tokens))
        
        if min_len == 0:
            return {'exact_match_accuracy': 0.0, 'prefix_accuracy': 0.0}
        
        # Apply attention masks if provided
        if reference_attention_mask is not None and hypothesis_attention_mask is not None:
            ref_mask = reference_attention_mask.cpu().numpy().astype(bool)
            hyp_mask = hypothesis_attention_mask.cpu().numpy().astype(bool)
            valid_positions = ref_mask[:min_len] & hyp_mask[:min_len]
            
            if valid_positions.sum() == 0:
                return {'exact_match_accuracy': 0.0, 'prefix_accuracy': 0.0}
                
            ref_tokens_valid = ref_tokens[:min_len][valid_positions]
            hyp_tokens_valid = hyp_tokens[:min_len][valid_positions]
        else:
            ref_tokens_valid = ref_tokens[:min_len]
            hyp_tokens_valid = hyp_tokens[:min_len]
        
        # Exact match accuracy (all tokens must match in order)
        exact_match = torch.all(ref_tokens_valid == hyp_tokens_valid).item()
        
        # Position-wise accuracy
        position_matches = (ref_tokens_valid == hyp_tokens_valid).float()
        positional_accuracy = position_matches.mean().item()
        
        return {
            'exact_match_accuracy': float(exact_match),
            'positional_accuracy': positional_accuracy
        }
    
    @torch.no_grad() 
    def compute_forward_coherence(self,
                                 generated_prefix: torch.Tensor,
                                 true_suffix: torch.Tensor,
                                 prefix_attention_mask: torch.Tensor = None,
                                 suffix_attention_mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Compute forward coherence metrics.
        
        Measures how well the forward model can predict the true suffix
        when conditioned on the generated prefix.
        
        Args:
            generated_prefix: Generated prefix tokens [prefix_len]
            true_suffix: True suffix tokens [suffix_len]
            prefix_attention_mask: Mask for valid prefix tokens [prefix_len]
            suffix_attention_mask: Mask for valid suffix tokens [suffix_len]
            
        Returns:
            Dictionary containing forward coherence metrics
        """
        device = generated_prefix.device
        
        # Concatenate prefix and suffix
        full_sequence = torch.cat([generated_prefix, true_suffix], dim=0)
        
        # Create attention mask
        if prefix_attention_mask is not None and suffix_attention_mask is not None:
            full_attention_mask = torch.cat([prefix_attention_mask, suffix_attention_mask], dim=0)
        else:
            full_attention_mask = torch.ones_like(full_sequence)
        
        # Add batch dimension
        input_ids = full_sequence.unsqueeze(0)  # [1, seq_len]
        attention_mask = full_attention_mask.unsqueeze(0)  # [1, seq_len]
        
        # Forward pass through the model
        outputs = self.forward_model.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # We want to compute the perplexity of the suffix tokens
        prefix_len = len(generated_prefix)
        suffix_len = len(true_suffix)
        
        if suffix_len == 0:
            return {'forward_coherence_ppl': float('inf'), 'forward_coherence_loss': float('inf')}
        
        # Extract logits and labels for suffix prediction
        # Logits for predicting suffix: positions [prefix_len-1:prefix_len+suffix_len-1]
        # Labels for suffix: positions [prefix_len:prefix_len+suffix_len]
        suffix_logits = logits[0, prefix_len-1:prefix_len+suffix_len-1, :]  # [suffix_len, vocab_size]
        suffix_labels = true_suffix  # [suffix_len]
        suffix_mask = suffix_attention_mask if suffix_attention_mask is not None else torch.ones_like(true_suffix)
        
        # Compute loss
        loss_fct = CrossEntropyLoss(reduction='none')
        losses = loss_fct(suffix_logits, suffix_labels)  # [suffix_len]
        
        # Apply mask and compute average loss
        masked_losses = losses * suffix_mask.float()
        avg_loss = masked_losses.sum() / suffix_mask.sum() if suffix_mask.sum() > 0 else torch.tensor(float('inf'))
        
        # Compute perplexity
        perplexity = torch.exp(avg_loss)
        
        return {
            'forward_coherence_ppl': perplexity.item(),
            'forward_coherence_loss': avg_loss.item()
        }
    
    def compute_comprehensive_metrics(self,
                                      reference_prefix: torch.Tensor,
                                      generated_prefix: torch.Tensor,
                                      true_suffix: torch.Tensor,
                                      reference_prefix_mask: torch.Tensor,
                                      generated_prefix_mask: torch.Tensor,
                                      suffix_mask: torch.Tensor,
                                      predicted_overall_text: str,
                                      original_overall_text: str,
                                      suffix_text: str,
                                      ) -> Dict[str, Any]:
        """
        Compute all evaluation metrics for a single sample.
        
        Args:
            reference_prefix: Ground truth prefix tokens
            generated_prefix: Generated prefix tokens
            true_suffix: True suffix tokens
            reference_prefix_mask: Mask for reference prefix
            generated_prefix_mask: Mask for generated prefix
            suffix_mask: Mask for suffix
            predicted_overall_text: Predicted overall text
            original_overall_text: Original overall text
            suffix_text: Suffix text
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Token overlap metrics
        overlap_metrics = self.compute_token_overlap_metrics(
            reference_prefix, generated_prefix,
            reference_prefix_mask, generated_prefix_mask
        )
        metrics.update(overlap_metrics)

        # Sentences perplexities computed by a third-party model
        ppl_metrics = self.compute_third_party_perplexity(
            predicted_overall_text, original_overall_text, suffix_text,
        )
        metrics.update(ppl_metrics)
        
        # Positional accuracy metrics
        positional_metrics = self.compute_positional_accuracy(
            reference_prefix, generated_prefix,
            reference_prefix_mask, generated_prefix_mask
        )
        metrics.update(positional_metrics)
        
        # Forward coherence metrics
        coherence_metrics = self.compute_forward_coherence(
            generated_prefix, true_suffix,
            generated_prefix_mask, suffix_mask
        )
        metrics.update(coherence_metrics)

        # Semantic similarity metrics
        semantic_metrics = self.compute_semantic_similarity(
            original_overall_text, predicted_overall_text
        )
        metrics.update(semantic_metrics)

        # Token duplication metrics
        token_duplication_metrics = self.compute_token_duplications(generated_prefix)
        metrics.update(token_duplication_metrics)
        
        return metrics

    def compute_third_party_perplexity(self, predicted_text: str, original_text: str, suffix_text: str) -> Dict[
        str, float]:
        """
        Compute perplexity metrics using a third-party model.

        This function is a placeholder for integrating with an external model
        that computes perplexity based on the predicted and original texts.

        Args:
            predicted_text: The text generated by the model
            original_text: The original text to compare against
            suffix_text: The suffix text to consider for perplexity

        Returns:
            Dictionary containing perplexity metrics
        """
        # Limit the length of the texts to avoid excessive computation
        # Leave only the first 50 words
        predicted_text = ' '.join(predicted_text.split()[:50])
        original_text = ' '.join(original_text.split()[:50])
        full_predicted_sentence = f"{predicted_text} {suffix_text}"
        full_original_sentence = f"{original_text} {suffix_text}"
        sentences = [predicted_text, original_text, full_predicted_sentence, full_original_sentence]

        predicted_ppl, original_ppl, full_predicted_ppl, full_original_ppl = self.perplexity_evaluator.compute(
            model_id=self.external_llm,
            predictions=sentences,
            add_start_token=False,  # Set to True if you want to include the probability of the first token
        )['perplexities']

        return {
            'predicted_prefix_perplexity': predicted_ppl,
            'original_prefix_perplexity': original_ppl,
            'full_predicted_perplexity': full_predicted_ppl,
            'full_original_perplexity': full_original_ppl,
        }

    def compute_semantic_similarity(self, reference_text: str,
                                    generated_text: str) -> dict[str, float]:

        """
        Compute semantic similarity metrics between reference and generated texts.

        Args:
            reference_text: Ground truth text
            generated_text: Generated text

        Returns:
            Dictionary containing semantic similarity metrics
        """
        # Use a pre-trained model for semantic similarity
        if not reference_text.strip() or not generated_text.strip():
            return {
                'semantic_similarity': 0.0,
            }

        embeddings = self.semantic_model.encode([reference_text, generated_text], show_progress_bar=False)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return {
            'semantic_similarity': float(similarity),
        }

    def compute_token_duplications(self, generated_tokens: torch.Tensor) -> dict[str, int]:
        """
        Compute token duplication metrics in the generated prefix.

        Args:
            generated_tokens: Generated prefix tokens [seq_len]
        Returns:
            Number of duplicated tokens in the generated prefix
        """
        # Convert to CPU for easier processing
        tokens = generated_tokens.cpu().numpy()

        # Count unique tokens
        unique_tokens = set(tokens)

        # Count total tokens
        total_tokens = len(tokens)

        # Calculate duplicates
        num_duplicates = total_tokens - len(unique_tokens)

        return {
            'token_duplications': num_duplicates,
        }



def aggregate_metrics(sample_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        sample_metrics: List of metric dictionaries from individual samples
        
    Returns:
        Dictionary of aggregated metrics (means, std, etc.)
    """
    if not sample_metrics:
        return {}
    
    # Collect all metric keys
    all_keys = set()
    for metrics in sample_metrics:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in sample_metrics if key in metrics]
        
        if values:
            values = np.array(values)
            aggregated[f'mean_{key}'] = np.mean(values)
            aggregated[f'std_{key}'] = np.std(values)
    
    return aggregated


# Helper functions for integration with existing code

def extract_prefix_suffix_from_sequence(input_ids: torch.Tensor,
                                       attention_mask: torch.Tensor,
                                       prefix_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract prefix and suffix from a full sequence.
    
    Args:
        input_ids: Full sequence tokens [seq_len]
        attention_mask: Attention mask [seq_len]
        prefix_length: Length of prefix to extract
        
    Returns:
        Tuple of (prefix_tokens, suffix_tokens, prefix_mask, suffix_mask)
    """
    seq_len = input_ids.size(0)
    
    if prefix_length >= seq_len:
        # Edge case: prefix is longer than sequence
        prefix_tokens = input_ids
        suffix_tokens = torch.tensor([], dtype=input_ids.dtype, device=input_ids.device)
        prefix_mask = attention_mask
        suffix_mask = torch.tensor([], dtype=attention_mask.dtype, device=attention_mask.device)
    else:
        prefix_tokens = input_ids[:prefix_length]
        suffix_tokens = input_ids[prefix_length:]
        prefix_mask = attention_mask[:prefix_length] 
        suffix_mask = attention_mask[prefix_length:]
    
    return prefix_tokens, suffix_tokens, prefix_mask, suffix_mask
