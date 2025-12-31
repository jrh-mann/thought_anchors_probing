"""
Activation extraction using nnsight.
Based on reference implementation pattern.
"""

import nnsight
import torch
import json
import os
import gc
import logging
import threading
import queue
from typing import List, Dict, Optional, Any
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utils import reconstruct_sequence_messages, format_with_chat_template

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_last_content_token_positions(token_ids: torch.Tensor, tokenizer: PreTrainedTokenizer, num_tokens: int = 1) -> List[int]:
    """
    Find the last N non-special token positions by walking backwards from the end.
    
    Args:
        token_ids: Tensor of token IDs
        tokenizer: Tokenizer to identify special tokens
        num_tokens: Number of content token positions to return (default 1)
    
    Returns:
        List of indices for the last N content tokens, in forward order (earliest first).
        If fewer than num_tokens content tokens exist, returns as many as found.
    """
    special_ids = set(tokenizer.all_special_ids)
    seq_len = len(token_ids)
    
    # Walk backwards from the end, collecting non-special token positions
    content_positions = []
    for pos in range(seq_len - 1, -1, -1):
        token_id = token_ids[pos].item()
        if token_id not in special_ids:
            content_positions.append(pos)
            if len(content_positions) >= num_tokens:
                break
    
    # Reverse to get forward order (earliest position first)
    content_positions.reverse()
    
    # Fallback: if no content tokens found, return [-2]
    if not content_positions:
        return [seq_len - 2] if seq_len >= 2 else [0]
    
    return content_positions


def find_last_content_token_position(token_ids: torch.Tensor, tokenizer: PreTrainedTokenizer) -> int:
    """
    Find the last non-special token position by walking backwards from the end.
    
    DEPRECATED: Use find_last_content_token_positions(token_ids, tokenizer, num_tokens=1) instead.
    Kept for backward compatibility.
    
    Args:
        token_ids: Tensor of token IDs
        tokenizer: Tokenizer to identify special tokens
    
    Returns:
        Index of the last content token (fallback to -2 if not found)
    """
    positions = find_last_content_token_positions(token_ids, tokenizer, num_tokens=1)
    return positions[-1]


def _save_activations_worker(save_queue: queue.Queue) -> None:
    """Worker function to save activations from queue to disk."""
    while True:
        item = save_queue.get()
        if item is None:  # Sentinel to stop worker
            break
        
        save_path, token_ids, activations = item
        try:
            # Save as tuple of (token_ids, activations)
            torch.save((token_ids, activations), save_path)
            logger.info(f"Saved activations to {save_path}")
        except Exception as e:
            logger.error(f"Error saving activations to {save_path}: {e}")
        finally:
            save_queue.task_done()


def store_activations(
    model_name: str,
    sentence_data_list: List[Dict[str, Any]],
    activations_dir: str,
    tokenizer: PreTrainedTokenizer,
    model: Any,  # nnsight.LanguageModel
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
    token_position: str = "final",
    num_tokens: int = 3,
    verbose: bool = False,
    full_rollout_text: Optional[str] = None
) -> None:
    """
    Extract and store activations from sentence data using single-pass optimization.
    
    Runs ONE forward pass on the full rollout and extracts activations for each 
    sentence by finding token positions via character offset mapping.
    
    Args:
        model_name: HuggingFace model identifier (for logging)
        sentence_data_list: List of sentence data dictionaries from rollout JSON
        activations_dir: Directory to save activation tensors
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded nnsight.LanguageModel
        start_idx: Starting sentence index (defaults to 0)
        end_idx: Ending sentence index (defaults to None, processes all)
        layer_idx: Specific layer index to extract (defaults to None, extracts all layers)
        token_position: Which token to extract (defaults to "final", can be "final" or integer index)
        num_tokens: Number of final content tokens to average over (defaults to 3).
        verbose: Print detailed output for review
        full_rollout_text: Full rollout text (REQUIRED for single-pass optimization)
    """
    import sys
    from pathlib import Path as PathLib
    sys.path.insert(0, str(PathLib(__file__).parent.parent))
    from utils.text_utils import split_sentences
    
    os.makedirs(activations_dir, exist_ok=True)
    logger.info(f"Activations will be saved to: {activations_dir}")
    
    if full_rollout_text is None:
        raise ValueError("full_rollout_text is required for optimized extraction")
    
    # Get formatted prompt from first sentence
    formatted_prompt = sentence_data_list[0].get("formatted_prompt", [])
    if not formatted_prompt:
        raise ValueError("formatted_prompt not found in sentence data")
    
    # Split rollout into sentences
    sentences = split_sentences(full_rollout_text)
    logger.info(f"Split rollout into {len(sentences)} sentences")
    
    # Determine which sentences to process
    end_idx = end_idx if end_idx is not None else len(sentence_data_list)
    sentences_to_process = sentence_data_list[start_idx:end_idx]
    sentence_indices = [s.get("sentence_index", i) for i, s in enumerate(sentences_to_process, start_idx)]
    logger.info(f"Processing {len(sentences_to_process)} sentences: {sentence_indices}")
    
    # Format full rollout (add_generation_prompt=False since content is complete)
    full_messages = formatted_prompt.copy()
    full_messages.append({"role": "assistant", "content": full_rollout_text})
    full_formatted = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    
    # Tokenize with offset mapping
    encoding = tokenizer(full_formatted, return_offsets_mapping=True, return_tensors='pt')
    full_tokens = encoding['input_ids'][0]
    offset_mapping = encoding['offset_mapping'][0].tolist()
    
    logger.info(f"Full sequence: {len(full_tokens)} tokens, {len(full_formatted)} chars")
    
    # Find where assistant content starts
    assistant_start = full_formatted.find(full_rollout_text)
    if assistant_start == -1:
        raise ValueError("Could not find rollout text in formatted string")
    
    # Pre-compute cumulative sentence lengths for position calculation
    cumulative_lens = [0]
    for sent in sentences:
        cumulative_lens.append(cumulative_lens[-1] + len(sent))
    
    # Build map: sentence_index -> token position of period
    special_ids = set(tokenizer.all_special_ids)
    
    def find_content_positions(sentence_idx: int) -> List[int]:
        """Find last N content token positions for a sentence."""
        sent = sentences[sentence_idx]
        
        # Find the period position in this sentence
        period_pos = sent.rfind('.')
        if period_pos == -1:
            period_pos = len(sent.rstrip()) - 1
        
        # Map to position in full formatted string
        target_char = assistant_start + cumulative_lens[sentence_idx] + period_pos
        
        # Find token containing this character
        token_pos = None
        for tok_idx, (start, end) in enumerate(offset_mapping):
            if start <= target_char < end:
                token_pos = tok_idx
                break
        
        if token_pos is None:
            logger.warning(f"Could not find token for sentence {sentence_idx}, using fallback")
            token_pos = len(full_tokens) - 2
        
        # Walk backwards to get last N content tokens
        positions = []
        for pos in range(token_pos, -1, -1):
            if full_tokens[pos].item() not in special_ids:
                positions.append(pos)
                if len(positions) >= num_tokens:
                    break
        positions.reverse()
        return positions if positions else [token_pos]
    
    # Run SINGLE forward pass on full rollout
    logger.info("Running single forward pass on full rollout...")
    gc.collect()
    torch.cuda.empty_cache()
    
    # nnsight requires .save() to access tensors outside the trace context
    saved_outputs = []
    with torch.no_grad():
        with model.trace(full_formatted):
            if layer_idx is not None:
                num_layers_model = len(model.model.layers)
                if layer_idx >= num_layers_model or layer_idx < 0:
                    raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers_model})")
                saved_outputs.append(model.model.layers[layer_idx].output[0].save())
            else:
                for layer in model.model.layers:
                    saved_outputs.append(layer.output[0].save())
    
    # Stack saved outputs (they are now accessible after trace completes)
    layer_tensors = [out for out in saved_outputs]
    all_activations = torch.stack(layer_tensors, dim=0).cpu()
    
    logger.info(f"Activations shape: {all_activations.shape}")
    
    # Extract and save activations for each sentence
    for sentence_data in sentences_to_process:
        sentence_idx = sentence_data.get("sentence_index", sentence_indices[0])
        
        if sentence_idx >= len(sentences):
            logger.warning(f"Sentence index {sentence_idx} >= {len(sentences)}, skipping")
            continue
        
        content_positions = find_content_positions(sentence_idx)
        
        # Extract and average activations at these positions
        token_acts = all_activations[:, content_positions, :]
        
        if verbose:
            # Get the exact sentence text
            sent_text = sentences[sentence_idx]
            
            # Calculate the char position of the sentence end in the formatted string
            sentence_end_char = assistant_start + cumulative_lens[sentence_idx + 1]
            
            # Show last 500 chars of the formatted string up to this sentence's end
            context_start = max(0, sentence_end_char - 500)
            context_snippet = full_formatted[context_start:sentence_end_char]
            
            # Get the extracted tokens
            tokens_text = [tokenizer.decode([full_tokens[p].item()]) for p in content_positions]
            
            print("\n" + "-" * 70)
            print(f"SENTENCE {sentence_idx} DEBUG:")
            print("-" * 70)
            print(f"  Sentence text ({len(sent_text)} chars):")
            print(f"    \"{sent_text[:200]}{'...' if len(sent_text) > 200 else ''}\"")
            print(f"\n  Token positions: {content_positions}")
            print(f"  Extracted tokens: {tokens_text}")
            print(f"\n  LAST 500 CHARS OF FORMATTED CONTEXT (ending at sentence {sentence_idx}):")
            print("  " + "~" * 48)
            # Print with indentation, showing where we are in the context
            for line in context_snippet.split('\n'):
                print(f"  {line}")
            print("  " + "~" * 48)
            print(f"  [Sentence ends at char {sentence_end_char} of {len(full_formatted)}]")
            print("-" * 70)
        
        # Save
        save_path = Path(activations_dir) / f"sentence_{sentence_idx}.pt"
        torch.save((full_tokens.cpu(), token_acts), save_path)
        logger.info(f"Saved sentence {sentence_idx}")
    
    logger.info("Activation storage complete!")