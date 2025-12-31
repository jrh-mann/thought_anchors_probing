#!/usr/bin/env python3
"""
Main script to extract activations from rollout JSON files using nnsight.

Supports both single file and batch (directory) modes.

Usage:
    # Single file
    python extract_activations.py --rollout-file src/rollouts/sat_solver_rollout_0_rollouts.json
    
    # Batch mode - process all rollout files in a directory
    python extract_activations.py --rollout-dir src/rollouts --output-dir workspace/activations
    
    # Test mode with verbose output
    python extract_activations.py --rollout-file src/rollouts/sat_solver_rollout_0_rollouts.json --test --debug
"""

import sys
import json
import argparse
import glob
import gc
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import nnsight
from transformers import AutoTokenizer
from activations.extract import store_activations
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEBUG = True  # Set to True for verbose debug output


def debug_print(msg: str, indent: int = 0) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        prefix = "  " * indent
        print(f"{prefix}{msg}")


def extract_prompt_name(rollout_file: str) -> str:
    """
    Extract prompt name from rollout filename.
    
    Example: 'src/rollouts/sat_solver_rollout_0_rollouts.json' -> 'sat_solver_rollout_0'
    """
    filename = Path(rollout_file).stem
    if filename.endswith('_rollouts'):
        return filename[:-9]  # Remove '_rollouts' suffix
    return filename


def load_rollout_json(rollout_file: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Load rollout JSON file, handling both old and new formats.
    
    New format (dict):
        {"full_rollout_text": "...", "sentence_data": [...]}
    
    Old format (list):
        [{"sentence_index": 0, ...}, ...]
    
    Returns:
        Tuple of (sentence_data_list, full_rollout_text)
        full_rollout_text is None for old format
    """
    debug_print(f"Loading: {rollout_file}")
    
    with open(rollout_file, 'r') as f:
        data = json.load(f)
    
    # Handle new format (dict with full_rollout_text and sentence_data)
    if isinstance(data, dict):
        if "sentence_data" in data:
            sentence_data = data["sentence_data"]
            full_rollout_text = data.get("full_rollout_text")
            
            debug_print(f"  Format: NEW (dict with full_rollout_text)", indent=1)
            debug_print(f"  Sentences: {len(sentence_data)}", indent=1)
            if full_rollout_text:
                debug_print(f"  full_rollout_text: {len(full_rollout_text)} chars", indent=1)
            
            return sentence_data, full_rollout_text
        else:
            raise ValueError(f"Dict format but no 'sentence_data' key found. Keys: {list(data.keys())}")
    
    # Handle old format (list of sentence data)
    elif isinstance(data, list):
        debug_print(f"  Format: OLD (list)", indent=1)
        debug_print(f"  Sentences: {len(data)}", indent=1)
        return data, None
    
    else:
        raise ValueError(f"Unexpected JSON format: {type(data)}")


def show_formatted_input(
    sentence_data_list: List[Dict[str, Any]],
    full_rollout_text: str,
    tokenizer: Any,
    sentence_idx: int = 0
) -> None:
    """
    Debug function to show how the input is reconstructed with Harmony tags.
    """
    if not DEBUG:
        return
    
    print("\n" + "=" * 70)
    print("DEBUG: RECONSTRUCTED INPUT WITH CHAT TEMPLATE")
    print("=" * 70)
    
    # Get formatted prompt from first sentence
    if not sentence_data_list:
        print("ERROR: No sentence data!")
        return
    
    formatted_prompt = sentence_data_list[0].get("formatted_prompt", [])
    if not formatted_prompt:
        print("ERROR: No formatted_prompt in sentence data!")
        return
    
    print(f"\n1. FORMATTED PROMPT (user messages):")
    print("-" * 50)
    for i, msg in enumerate(formatted_prompt):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:200]
        print(f"  [{i}] {role}: {content}...")
    
    print(f"\n2. FULL ROLLOUT TEXT (first 500 chars):")
    print("-" * 50)
    print(full_rollout_text[:500] if full_rollout_text else "None")
    
    # Reconstruct the full message and apply chat template
    print(f"\n3. APPLYING CHAT TEMPLATE:")
    print("-" * 50)
    
    full_messages = formatted_prompt.copy()
    full_messages.append({"role": "assistant", "content": full_rollout_text})
    
    try:
        full_formatted = tokenizer.apply_chat_template(
            full_messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        print(f"  Total formatted length: {len(full_formatted)} chars")
        print(f"\n  FORMATTED STRING (first 1000 chars):")
        print("  " + "-" * 48)
        # Print with indentation
        for line in full_formatted[:1000].split('\n'):
            print(f"  {line}")
        if len(full_formatted) > 1000:
            print(f"  ... [{len(full_formatted) - 1000} more chars]")
        
    except Exception as e:
        print(f"  ERROR applying chat template: {e}")
    
    print("\n" + "=" * 70 + "\n")


def save_metadata(
    prompt_name: str,
    sentence_data_list: List[Dict[str, Any]],
    activations_dir: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> None:
    """Save metadata JSON file for cross-referencing."""
    end_idx = end_idx if end_idx is not None else len(sentence_data_list)
    sentences_to_save = sentence_data_list[start_idx:end_idx]
    
    metadata = {
        'prompt_name': prompt_name,
        'sentence_data': [
            {
                'sentence_index': s.get('sentence_index', i),
                'sentence': s.get('sentence', ''),
                'p_reward_hacks': s.get('p_reward_hacks', 0.0),
                'file': f"sentence_{s.get('sentence_index', i)}.pt"
            }
            for i, s in enumerate(sentences_to_save)
        ]
    }
    
    metadata_path = Path(activations_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    debug_print(f"Saved metadata to {metadata_path}")


def find_rollout_files(rollout_dir: str) -> List[str]:
    """Find all rollout JSON files in a directory."""
    pattern = Path(rollout_dir) / "*_rollouts.json"
    files = sorted(glob.glob(str(pattern)))
    debug_print(f"Found {len(files)} rollout files in {rollout_dir}")
    return files


def process_single_rollout(
    rollout_file: str,
    output_dir: str,
    tokenizer: Any,
    model: Any,
    layer_idx: Optional[int] = None,
    num_tokens: int = 3,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    test_mode: bool = False,
    skip_if_exists: bool = True
) -> bool:
    """
    Process a single rollout file and extract activations.
    
    Returns:
        True if successful, False otherwise
    """
    prompt_name = extract_prompt_name(rollout_file)
    activations_dir = Path(output_dir) / prompt_name
    
    # Check if already processed
    metadata_path = activations_dir / "metadata.json"
    if skip_if_exists and metadata_path.exists():
        debug_print(f"  Skipping {prompt_name} (already exists)")
        return True
    
    print(f"\nProcessing: {prompt_name}")
    
    try:
        # Load rollout JSON (handles new format)
        sentence_data_list, full_rollout_text = load_rollout_json(rollout_file)
        
        if not sentence_data_list:
            print(f"  ERROR: No sentence data in {rollout_file}")
            return False
        
        if full_rollout_text is None:
            print(f"  ERROR: No full_rollout_text in {rollout_file} (required for optimized extraction)")
            return False
        
        # Debug: show formatted input for first file
        if DEBUG:
            show_formatted_input(sentence_data_list, full_rollout_text, tokenizer)
        
        # Adjust indices for test mode
        actual_start = start_idx
        actual_end = end_idx
        if test_mode:
            actual_start = 0
            actual_end = 1
            debug_print(f"  Test mode: processing only first sentence")
        
        # Create output directory
        activations_dir.mkdir(parents=True, exist_ok=True)
        
        debug_print(f"  Extracting activations for {len(sentence_data_list)} sentences...")
        debug_print(f"    Range: [{actual_start}:{actual_end if actual_end else 'end'}]")
        debug_print(f"    Num tokens per sentence: {num_tokens}")
        debug_print(f"    Layer: {layer_idx if layer_idx is not None else 'all'}")
        
        # Extract activations - NOW PASSING full_rollout_text!
        store_activations(
            model_name="openai/gpt-oss-20b",
            sentence_data_list=sentence_data_list,
            activations_dir=str(activations_dir),
            tokenizer=tokenizer,
            model=model,
            start_idx=actual_start,
            end_idx=actual_end,
            layer_idx=layer_idx,
            num_tokens=num_tokens,
            verbose=DEBUG,
            full_rollout_text=full_rollout_text  # ← THIS WAS MISSING!
        )
        
        # Save metadata
        save_metadata(
            prompt_name=prompt_name,
            sentence_data_list=sentence_data_list,
            activations_dir=str(activations_dir),
            start_idx=actual_start,
            end_idx=actual_end
        )
        
        debug_print(f"  ✓ Saved to {activations_dir}")
        return True
        
    except Exception as e:
        print(f"  ERROR processing {rollout_file}: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return False


def main():
    global DEBUG
    
    parser = argparse.ArgumentParser(
        description="Extract activations from rollout JSON files using nnsight"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rollout-file",
        type=str,
        help="Path to single rollout JSON file"
    )
    input_group.add_argument(
        "--rollout-dir",
        type=str,
        help="Directory containing rollout JSON files (batch mode)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/activations",
        help="Output directory (default: /workspace/activations)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name (default: openai/gpt-oss-20b)"
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Specific layer index to extract (default: None, extracts all layers)"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=3,
        help="Number of final tokens to extract per sentence (default: 3)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - process only first sentence of each rollout"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output (shows chat template, tokenization, etc.)"
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting sentence index (default: 0)"
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="Ending sentence index (default: None, processes all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip rollouts that already have activations (default: True)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Process all rollouts even if they already exist"
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping strategy (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for model (default: bfloat16)"
    )
    
    args = parser.parse_args()
    
    # Set debug mode
    DEBUG = args.debug
    
    # Set logging level
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Determine skip behavior
    skip_if_exists = args.skip_existing and not args.no_skip
    
    print("=" * 70)
    print("ACTIVATION EXTRACTION WITH NNSIGHT")
    print("=" * 70)
    print(f"\nConfiguration:")
    if args.rollout_file:
        print(f"  Mode: Single file")
        print(f"  Rollout file: {args.rollout_file}")
    else:
        print(f"  Mode: Batch (directory)")
        print(f"  Rollout dir: {args.rollout_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Layer index: {args.layer_idx if args.layer_idx is not None else 'all'}")
    print(f"  Num tokens: {args.num_tokens}")
    print(f"  Test mode: {args.test}")
    print(f"  Debug: {args.debug}")
    print(f"  Skip existing: {skip_if_exists}")
    print(f"  Start index: {args.start_idx}")
    print(f"  End index: {args.end_idx if args.end_idx is not None else 'all'}")
    print()
    
    # Get list of files to process
    if args.rollout_file:
        rollout_files = [args.rollout_file]
    else:
        rollout_files = find_rollout_files(args.rollout_dir)
        if not rollout_files:
            print(f"ERROR: No rollout files found in {args.rollout_dir}")
            return 1
    
    print(f"Files to process: {len(rollout_files)}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    debug_print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype.lower(), torch.bfloat16)
    
    # Load model with nnsight
    print(f"\nLoading model: {args.model_name}")
    print(f"  Device map: {args.device_map}")
    print(f"  Dtype: {dtype}")
    print("  This may take a minute to load the model...")
    
    model = nnsight.LanguageModel(
        args.model_name,
        device_map=args.device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    print("  ✓ Model loaded\n")
    
    # Process files
    start_time = time.time()
    successful = 0
    failed = 0
    skipped = 0
    
    for i, rollout_file in enumerate(rollout_files):
        print(f"\n[{i+1}/{len(rollout_files)}] ", end="")
        
        # Check if exists before processing
        prompt_name = extract_prompt_name(rollout_file)
        activations_dir = Path(args.output_dir) / prompt_name
        metadata_path = activations_dir / "metadata.json"
        
        if skip_if_exists and metadata_path.exists():
            print(f"Skipping {prompt_name} (exists)")
            skipped += 1
            continue
        
        success = process_single_rollout(
            rollout_file=rollout_file,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            model=model,
            layer_idx=args.layer_idx,
            num_tokens=args.num_tokens,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            test_mode=args.test,
            skip_if_exists=skip_if_exists
        )
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Cleanup periodically
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"\nActivations saved to: {args.output_dir}")
    print(f"\nTo load with quick_load:")
    print(f"  from src.activations.load import quick_load")
    print(f"  activations, metadata, labels = quick_load('{args.output_dir}')")
    print()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
