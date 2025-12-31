#!/usr/bin/env python3
"""
Extract negative sample activations from Andy Arditi's gpt-oss-20b rollouts dataset.

This script downloads diverse rollouts from HuggingFace, extracts activations at
random token positions, and saves them with label=0 to serve as negative samples
for reward-hacking probing.

Dataset: https://huggingface.co/datasets/andyrdt/gpt-oss-20b-rollouts

Configure parameters at the top of this file, then run:
    python scripts/extract_negative_activations.py
"""

import sys
from pathlib import Path
import os
import gc
import json
import random
import time
from typing import List, Dict, Optional, Any, Tuple

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Dataset configuration
DATASET_NAME = "andyrdt/gpt-oss-20b-rollouts"
SUBSET = "ultrachat_200k"  # None = "combined" (all subsets), or specific like "ultrachat_200k"
               # Options: "ultrachat_200k", "WildChat-1M", "oasst1", "gsm8k", 
               #          "HarmBench", "combined", etc.
SPLIT = None   # None = auto-detect first available split

# Sampling configuration
NUM_ROLLOUTS = 1000         # Number of rollouts to sample from dataset
TOKENS_PER_ROLLOUT = 10      # Random token positions to extract per rollout
MIN_ASSISTANT_TOKENS = 20   # Minimum assistant response length to include

# Model configuration
MODEL_NAME = "openai/gpt-oss-20b"
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.8

# Output configuration
OUTPUT_DIR = "/workspace/activations"
LAYER_IDX = None  # None = all layers, or specific layer index

# Reproducibility
SEED = 42

# Debug mode - verbose printing for monitoring
DEBUG = False

# ============================================================================
# END CONFIGURATION
# ============================================================================


def debug_print(msg: str, indent: int = 0) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        prefix = "  " * indent
        print(f"{prefix}{msg}")


def load_dataset_cached(
    dataset_name: str,
    subset: Optional[str] = None,
    split: Optional[str] = None
) -> Any:
    """
    Load dataset from HuggingFace with automatic caching.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        subset: Specific subset/config to load (None = combined/default)
        split: Specific split to load (None = auto-detect)
    
    Returns:
        HuggingFace dataset object
    """
    from datasets import load_dataset
    
    # Determine subset to use
    config_name = subset if subset else "combined"
    
    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"  Dataset: {dataset_name}")
    print(f"  Subset: {config_name}")
    
    debug_print(f"Loading from HuggingFace (will cache automatically)...")
    
    try:
        # Try loading with specified config
        ds = load_dataset(dataset_name, config_name, trust_remote_code=True)
        debug_print(f"Successfully loaded config: {config_name}")
    except Exception as e:
        debug_print(f"Could not load config '{config_name}': {e}")
        debug_print("Trying to load default config...")
        ds = load_dataset(dataset_name, trust_remote_code=True)
    
    # Get available splits
    if isinstance(ds, dict):
        available_splits = list(ds.keys())
        debug_print(f"Available splits: {available_splits}")
        
        # Select split
        if split is not None:
            if split not in available_splits:
                raise ValueError(f"Split '{split}' not found. Available: {available_splits}")
            selected_split = split
        else:
            # Auto-select first available split
            preferred_order = ["train", "train_sft", "test", "validation"]
            selected_split = None
            for pref in preferred_order:
                if pref in available_splits:
                    selected_split = pref
                    break
            if selected_split is None:
                selected_split = available_splits[0]
        
        debug_print(f"Using split: {selected_split}")
        ds = ds[selected_split]
    
    print(f"  Total rows: {len(ds):,}")
    print(f"  Columns: {ds.column_names}")
    print()
    
    return ds


def inspect_dataset_structure(ds: Any, num_samples: int = 3) -> None:
    """
    Print detailed information about dataset structure for debugging.
    
    Args:
        ds: HuggingFace dataset
        num_samples: Number of sample rows to display
    """
    if not DEBUG:
        return
    
    print("=" * 70)
    print("DATASET STRUCTURE INSPECTION")
    print("=" * 70)
    
    print(f"\nSchema:")
    for col in ds.column_names:
        # Get first non-null value to determine type
        sample_val = None
        for i in range(min(10, len(ds))):
            if ds[i][col] is not None:
                sample_val = ds[i][col]
                break
        
        val_type = type(sample_val).__name__ if sample_val is not None else "unknown"
        val_preview = ""
        if isinstance(sample_val, str):
            val_preview = f' (e.g., "{sample_val[:50]}...")'
        print(f"  - {col}: {val_type}{val_preview}")
    
    print(f"\nSample rows ({num_samples}):")
    for i in range(min(num_samples, len(ds))):
        row = ds[i]
        print(f"\n  Row {i}:")
        for col in ds.column_names:
            val = row[col]
            if isinstance(val, str):
                val_display = f'"{val[:80]}..."' if len(val) > 80 else f'"{val}"'
            else:
                val_display = str(val)
            print(f"    {col}: {val_display}")
    
    print()


def sample_rollouts(
    ds: Any,
    num_rollouts: int,
    min_assistant_tokens: int,
    seed: int
) -> List[Dict[str, Any]]:
    """
    Sample N rollouts from the dataset, filtering for minimum length.
    
    Args:
        ds: HuggingFace dataset
        num_rollouts: Number of rollouts to sample
        min_assistant_tokens: Minimum assistant response length
        seed: Random seed
    
    Returns:
        List of row dictionaries
    """
    print("=" * 70)
    print("SAMPLING ROLLOUTS")
    print("=" * 70)
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle indices
    all_indices = list(range(len(ds)))
    random.shuffle(all_indices)
    
    sampled = []
    skipped_short = 0
    skipped_missing = 0
    
    debug_print(f"Scanning dataset for valid rollouts (need {num_rollouts})...")
    
    for idx in all_indices:
        if len(sampled) >= num_rollouts:
            break
        
        row = ds[idx]
        
        # Check required fields exist
        user_content = row.get("user_content")
        assistant_thinking = row.get("assistant_thinking")
        assistant_content = row.get("assistant_content")
        
        if not user_content or not assistant_thinking:
            skipped_missing += 1
            continue
        
        # Check minimum length (rough token estimate: chars / 4)
        thinking_len = len(assistant_thinking) if assistant_thinking else 0
        content_len = len(assistant_content) if assistant_content else 0
        estimated_tokens = (thinking_len + content_len) // 4
        
        if estimated_tokens < min_assistant_tokens:
            skipped_short += 1
            continue
        
        sampled.append({
            "index": idx,
            "user_content": user_content,
            "system_reasoning_effort": row.get("system_reasoning_effort", "medium"),
            "assistant_thinking": assistant_thinking,
            "assistant_content": assistant_content or "",
        })
        
        if DEBUG and len(sampled) % 100 == 0:
            debug_print(f"  Sampled {len(sampled)}/{num_rollouts}...")
    
    print(f"  Sampled: {len(sampled)} rollouts")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Skipped (missing fields): {skipped_missing}")
    print()
    
    if len(sampled) < num_rollouts:
        print(f"  WARNING: Could only sample {len(sampled)}/{num_rollouts} rollouts")
    
    return sampled


def format_conversation(
    row: Dict[str, Any],
    tokenizer: Any
) -> Tuple[str, Dict[str, Any]]:
    """
    Format a rollout row using the model's chat template.
    
    Following the dataset's documented format exactly:
    https://huggingface.co/datasets/andyrdt/gpt-oss-20b-rollouts
    
    Args:
        row: Rollout row dictionary
        tokenizer: Model tokenizer
    
    Returns:
        Tuple of (formatted_string, metadata_dict)
    """
    # Build conversation in the format specified by the dataset
    conversation = [
        {"role": "user", "content": row["user_content"]},
        {
            "role": "assistant", 
            "content": row["assistant_content"],
            "thinking": row["assistant_thinking"]
        },
    ]
    
    # Apply chat template with reasoning_effort parameter
    reasoning_effort = row.get("system_reasoning_effort", "medium")
    
    try:
        formatted = tokenizer.apply_chat_template(
            conversation,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=False,
            tokenize=False,
        )
    except TypeError:
        # Fallback if reasoning_effort not supported
        debug_print("  Note: reasoning_effort parameter not supported, using default template")
        formatted = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
        )
    
    metadata = {
        "user_content_len": len(row["user_content"]),
        "thinking_len": len(row["assistant_thinking"]) if row["assistant_thinking"] else 0,
        "content_len": len(row["assistant_content"]) if row["assistant_content"] else 0,
        "reasoning_effort": reasoning_effort,
    }
    
    return formatted, metadata


def find_assistant_token_range(
    formatted_text: str,
    token_ids: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    assistant_thinking: str,
    tokenizer: Any
) -> Tuple[int, int]:
    """
    Find the token index range corresponding to the assistant response.
    
    gpt-oss-20b uses Harmony format with channel tags like:
    - <|start_header_id|>assistant<|end_header_id|>
    - <|reserved_special_token_N|> for channel markers
    
    Args:
        formatted_text: Full formatted conversation string
        token_ids: Token IDs tensor
        offset_mapping: Character offset mapping from tokenizer
        assistant_thinking: The assistant's thinking content
        tokenizer: Tokenizer for special token detection
    
    Returns:
        Tuple of (start_idx, end_idx) for assistant tokens
    """
    # Try multiple markers in order of preference for Harmony format
    # gpt-oss uses <|start_header_id|>assistant<|end_header_id|> pattern
    harmony_markers = [
        "<|start_header_id|>assistant<|end_header_id|>",  # Primary Harmony marker
        "<|end_header_id|>",  # After any header
        "assistant<|end_header_id|>",  # Assistant header end
    ]
    
    assistant_char_start = -1
    marker_found = None
    
    for marker in harmony_markers:
        pos = formatted_text.find(marker)
        if pos != -1:
            # Start after the marker
            assistant_char_start = pos + len(marker)
            marker_found = marker
            debug_print(f"  Found Harmony marker: '{marker}' at pos {pos}", indent=2)
            break
    
    if assistant_char_start == -1:
        # Fallback: try to find assistant thinking content directly
        if assistant_thinking:
            # Find the start of the thinking content
            think_start = formatted_text.find(assistant_thinking[:min(50, len(assistant_thinking))])
            if think_start != -1:
                assistant_char_start = think_start
                debug_print(f"  Found assistant thinking content at pos {think_start}", indent=2)
        
        if assistant_char_start == -1:
            # Last fallback: estimate based on text structure
            debug_print("  Warning: Could not find Harmony markers, using heuristic fallback", indent=2)
            assistant_char_start = len(formatted_text) // 3
    
    # Find token index for this character position
    start_token_idx = 0
    end_token_idx = len(token_ids) - 1
    
    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start <= assistant_char_start < char_end:
            start_token_idx = tok_idx
            break
        if char_start >= assistant_char_start:
            start_token_idx = tok_idx
            break
    
    # Skip special tokens at the end
    special_ids = set(tokenizer.all_special_ids)
    while end_token_idx > start_token_idx and token_ids[end_token_idx].item() in special_ids:
        end_token_idx -= 1
    
    return start_token_idx, end_token_idx


def select_random_positions(
    start_idx: int,
    end_idx: int,
    num_positions: int,
    token_ids: torch.Tensor,
    tokenizer: Any,
    seed: int
) -> List[int]:
    """
    Select random token positions within the assistant response, avoiding special tokens.
    
    Args:
        start_idx: Start of assistant response tokens
        end_idx: End of assistant response tokens
        num_positions: Number of positions to select
        token_ids: Token IDs tensor
        tokenizer: Tokenizer for special token detection
        seed: Random seed (combined with sample index for reproducibility)
    
    Returns:
        List of selected token positions
    """
    random.seed(seed)
    
    special_ids = set(tokenizer.all_special_ids)
    
    # Get all valid (non-special) positions in range
    valid_positions = []
    for pos in range(start_idx, end_idx + 1):
        if token_ids[pos].item() not in special_ids:
            valid_positions.append(pos)
    
    if len(valid_positions) == 0:
        debug_print(f"  Warning: No valid positions found in range [{start_idx}, {end_idx}]")
        return []
    
    # Sample without replacement if possible
    num_to_sample = min(num_positions, len(valid_positions))
    selected = random.sample(valid_positions, num_to_sample)
    
    return sorted(selected)


def extract_activations_at_positions(
    formatted_text: str,
    positions: List[int],
    model: Any,
    layer_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Run forward pass and extract activations at specified positions.
    
    Args:
        formatted_text: Formatted conversation string
        positions: Token positions to extract
        model: nnsight.LanguageModel
        layer_idx: Specific layer to extract (None = all layers)
    
    Returns:
        Activations tensor of shape (num_layers, num_positions, hidden_dim)
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run forward pass with nnsight
    saved_outputs = []
    with torch.no_grad():
        with model.trace(formatted_text):
            if layer_idx is not None:
                num_layers_model = len(model.model.layers)
                if layer_idx >= num_layers_model or layer_idx < 0:
                    raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers_model})")
                saved_outputs.append(model.model.layers[layer_idx].output[0].save())
            else:
                for layer in model.model.layers:
                    saved_outputs.append(layer.output[0].save())
    
    # Stack layer outputs: (num_layers, seq_len, hidden_dim)
    all_activations = torch.stack([out.cpu() for out in saved_outputs], dim=0)
    
    # Extract at specified positions: (num_layers, num_positions, hidden_dim)
    # Remove batch dimension if present
    if all_activations.dim() == 4:
        all_activations = all_activations.squeeze(1)  # (num_layers, seq_len, hidden_dim)
    
    position_activations = all_activations[:, positions, :]
    
    return position_activations


def save_sample_activations(
    sample_idx: int,
    token_ids: torch.Tensor,
    positions: List[int],
    activations: torch.Tensor,
    metadata: Dict[str, Any],
    output_dir: str,
    tokenizer: Any
) -> None:
    """
    Save activations for a single sample in format compatible with quick_load.
    
    Creates structure that matches existing pipeline:
    - Directory: negative_{sample_idx}/
    - Files: sentence_{position_idx}.pt for each position
    - metadata.json with sentence_data array
    
    Args:
        sample_idx: Sample index
        token_ids: Full sequence token IDs
        positions: Token positions that were extracted
        activations: Activation tensor (num_layers, num_positions, hidden_dim)
        metadata: Metadata dictionary
        output_dir: Output directory
        tokenizer: Tokenizer for decoding token text
    """
    # Create directory named like a "prompt" for quick_load compatibility
    sample_dir = Path(output_dir) / f"negative_{sample_idx}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Build sentence_data array for metadata.json
    sentence_data = []
    
    for pos_idx, token_pos in enumerate(positions):
        # Extract activation for this position: (num_layers, 1, hidden_dim)
        pos_activation = activations[:, pos_idx:pos_idx+1, :]
        
        # Get token text for this position
        token_text = tokenizer.decode([token_ids[token_pos].item()])
        
        # Save as sentence_{pos_idx}.pt in format (token_ids, activations)
        # Note: we save the full token_ids for context, but activations are just this position
        pt_path = sample_dir / f"sentence_{pos_idx}.pt"
        torch.save((token_ids.cpu(), pos_activation.cpu()), pt_path)
        
        # Add to sentence_data array
        sentence_data.append({
            "sentence_index": pos_idx,
            "sentence": token_text,  # The token at this position
            "p_reward_hacks": 0.0,   # Negative sample = 0.0
            "token_position": token_pos,  # Original position in sequence (for reference)
        })
    
    # Create metadata.json matching quick_load expectations
    metadata_json = {
        "sentence_data": sentence_data,
        # Additional metadata for context
        "sample_type": "negative",
        "source_dataset": metadata.get("subset", "unknown"),
        "original_dataset_idx": metadata.get("original_dataset_idx"),
        "user_content_len": metadata.get("user_content_len"),
        "thinking_len": metadata.get("thinking_len"),
        "content_len": metadata.get("content_len"),
        "reasoning_effort": metadata.get("reasoning_effort"),
        "num_layers": activations.shape[0],
        "hidden_dim": activations.shape[2],
    }
    
    metadata_path = sample_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f, indent=2)


def main():
    """Main extraction pipeline."""
    
    print("=" * 70)
    print("NEGATIVE SAMPLES ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Subset: {SUBSET or 'combined (all)'}")
    print(f"  Num rollouts: {NUM_ROLLOUTS}")
    print(f"  Tokens per rollout: {TOKENS_PER_ROLLOUT}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Layer idx: {LAYER_IDX or 'all'}")
    print(f"  Seed: {SEED}")
    print(f"  Debug mode: {DEBUG}")
    print()
    
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # =========================================================================
    # Step 1: Load dataset
    # =========================================================================
    ds = load_dataset_cached(DATASET_NAME, SUBSET, SPLIT)
    
    if DEBUG:
        inspect_dataset_structure(ds)
    
    # =========================================================================
    # Step 2: Sample rollouts
    # =========================================================================
    sampled_rollouts = sample_rollouts(ds, NUM_ROLLOUTS, MIN_ASSISTANT_TOKENS, SEED)
    
    if not sampled_rollouts:
        print("ERROR: No rollouts could be sampled!")
        return 1
    
    # =========================================================================
    # Step 3: Initialize model and tokenizer
    # =========================================================================
    print("=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)
    
    from transformers import AutoTokenizer
    import nnsight
    
    debug_print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    debug_print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    debug_print(f"Loading model with nnsight: {MODEL_NAME}")
    debug_print(f"  This may take a few minutes...")
    
    model = nnsight.LanguageModel(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Get model info
    try:
        num_layers = len(model.model.layers)
        hidden_size = model.config.hidden_size
        debug_print(f"  Model loaded: {num_layers} layers, hidden_size={hidden_size}")
    except Exception as e:
        debug_print(f"  Model loaded (could not get layer info: {e})")
    
    print()
    
    # =========================================================================
    # Step 4: Process rollouts and extract activations
    # =========================================================================
    print("=" * 70)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, rollout in enumerate(sampled_rollouts):
        sample_seed = SEED + i  # Unique seed per sample for position selection
        
        print(f"\n[{i+1}/{len(sampled_rollouts)}] Processing rollout (original idx: {rollout['index']})...")
        
        try:
            # Format conversation
            debug_print(f"Formatting conversation...", indent=1)
            formatted_text, format_metadata = format_conversation(rollout, tokenizer)
            debug_print(f"User content: {format_metadata['user_content_len']} chars", indent=2)
            debug_print(f"Assistant thinking: {format_metadata['thinking_len']} chars", indent=2)
            debug_print(f"Assistant content: {format_metadata['content_len']} chars", indent=2)
            debug_print(f"Formatted total: {len(formatted_text)} chars", indent=2)
            
            # Print first 500 chars of formatted string for debugging chat template
            if DEBUG:
                preview = formatted_text[:1000]
                print(f"\n{'='*60}")
                print(f"FORMATTED STRING PREVIEW (first 500 chars):")
                print(f"{'='*60}")
                print(preview)
                print(f"{'='*60}\n")
            
            # Tokenize
            debug_print(f"Tokenizing...", indent=1)
            encoding = tokenizer(
                formatted_text, 
                return_offsets_mapping=True, 
                return_tensors='pt'
            )
            token_ids = encoding['input_ids'][0]
            offset_mapping = encoding['offset_mapping'][0].tolist()
            debug_print(f"Total tokens: {len(token_ids)}", indent=2)
            
            # Find assistant token range
            debug_print(f"Finding assistant token range...", indent=1)
            start_idx, end_idx = find_assistant_token_range(
                formatted_text, 
                token_ids, 
                offset_mapping,
                rollout["assistant_thinking"],
                tokenizer
            )
            debug_print(f"Assistant range: [{start_idx}, {end_idx}] ({end_idx - start_idx + 1} tokens)", indent=2)
            
            # Select random positions
            debug_print(f"Selecting {TOKENS_PER_ROLLOUT} random positions...", indent=1)
            positions = select_random_positions(
                start_idx, end_idx, TOKENS_PER_ROLLOUT,
                token_ids, tokenizer, sample_seed
            )
            
            if not positions:
                debug_print(f"No valid positions found, skipping", indent=2)
                failed += 1
                continue
            
            debug_print(f"Selected positions: {positions}", indent=2)
            
            # Show tokens at selected positions
            if DEBUG:
                tokens_text = [tokenizer.decode([token_ids[p].item()]) for p in positions]
                debug_print(f"Tokens at positions: {tokens_text[:5]}{'...' if len(tokens_text) > 5 else ''}", indent=2)
            
            # Extract activations
            debug_print(f"Running forward pass...", indent=1)
            activations = extract_activations_at_positions(
                formatted_text, positions, model, LAYER_IDX
            )
            debug_print(f"Activations shape: {tuple(activations.shape)}", indent=2)
            
            # Save
            debug_print(f"Saving to {OUTPUT_DIR}/sample_{i}/", indent=1)
            metadata = {
                **format_metadata,
                "original_dataset_idx": rollout["index"],
                "subset": SUBSET or "combined",
            }
            save_sample_activations(i, token_ids, positions, activations, metadata, OUTPUT_DIR, tokenizer)
            
            successful += 1
            debug_print(f"✓ Sample {i} complete", indent=1)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            failed += 1
            continue
        
        # Periodic cleanup
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Avg time per sample: {elapsed/max(successful,1):.1f} seconds")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"Total activations: {successful * TOKENS_PER_ROLLOUT} token positions")
    
    # Save global metadata
    global_metadata = {
        "dataset": DATASET_NAME,
        "subset": SUBSET or "combined",
        "num_rollouts_requested": NUM_ROLLOUTS,
        "num_rollouts_processed": successful,
        "tokens_per_rollout": TOKENS_PER_ROLLOUT,
        "total_positions": successful * TOKENS_PER_ROLLOUT,
        "model": MODEL_NAME,
        "layer_idx": LAYER_IDX,
        "seed": SEED,
        "label": 0,  # All negative samples
    }
    
    with open(Path(OUTPUT_DIR) / "global_metadata.json", "w") as f:
        json.dump(global_metadata, f, indent=2)
    
    print(f"\nGlobal metadata saved to: {OUTPUT_DIR}/global_metadata.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

