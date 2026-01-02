#!/usr/bin/env python3
"""
Extract baseline activations with smeared binary labels from counterfactual rollouts.

This script uses the same counterfactual continuations as the counterfactual probing method,
but applies a "smeared" binary label (0 or 1) to ALL sampled tokens based on the outcome
(whether the counterfactual contains "expected.json" indicating reward hacking).

This provides a fair comparison baseline - same data source, different labeling strategy.

Usage:
    python scripts/extract_baseline_activations.py
    python scripts/extract_baseline_activations.py --debug
    python scripts/extract_baseline_activations.py --max-samples 10000 --debug
"""

import sys
from pathlib import Path
import os
import gc
import json
import random
import time
import threading
import queue
from typing import List, Dict, Optional, Any, Tuple

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.reward_detector import detect_reward_hacking

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Input configuration
ROLLOUTS_DIR = Path(__file__).parent.parent / "src" / "rollouts"

# Sampling configuration
MAX_SAMPLES = 50000           # Maximum number of samples to extract (for fair comparison)
ROLLOUTS_PER_SENTENCE = 2     # Only use first N of 50 rollouts per sentence
TOKENS_PER_SAMPLE = 10        # Random token positions to extract per reconstructed text

# Model configuration
MODEL_NAME = "openai/gpt-oss-20b"

# Output configuration
OUTPUT_DIR = "/workspace/activations2"
LAYER_IDX = None  # None = all layers, or specific layer index

# Reproducibility
SEED = 42

# Debug mode - verbose printing for monitoring
DEBUG = False

# Perf logging (optional)
PERF_LOG = False
PERF_LOG_EVERY = 50  # batches
PERF_PUT_WARN_S = 0.25
PERF_SAVE_WARN_S = 0.50

# ============================================================================
# END CONFIGURATION
# ============================================================================


def debug_print(msg: str, indent: int = 0) -> None:
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        prefix = "  " * indent
        print(f"{prefix}{msg}")


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.1f}ms"


class _PerfEMA:
    """Simple exponential moving average for low-overhead perf logging."""

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = float(alpha)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * float(x)
        return self.value


class _PerfCounters:
    def __init__(self) -> None:
        self.batch_count = 0
        self.ema_tokenize_s = _PerfEMA()
        self.ema_forward_s = _PerfEMA()
        self.ema_index_s = _PerfEMA()
        self.ema_cpu_copy_s = _PerfEMA()
        self.ema_put_block_s = _PerfEMA()
        self.ema_empty_cache_s = _PerfEMA()
        self.ema_batch_total_s = _PerfEMA()

    def maybe_print(
        self,
        *,
        every: int,
        queue_size: int,
        queue_max: int,
        last_batch_items: int,
        last_batch_max_seq_len: int,
    ) -> None:
        if every <= 0:
            return
        if self.batch_count % every != 0:
            return
        tok = self.ema_tokenize_s.value or 0.0
        fwd = self.ema_forward_s.value or 0.0
        idx = self.ema_index_s.value or 0.0
        cpy = self.ema_cpu_copy_s.value or 0.0
        put = self.ema_put_block_s.value or 0.0
        ec = self.ema_empty_cache_s.value or 0.0
        tot = self.ema_batch_total_s.value or 0.0
        print(
            "[PERF] "
            f"batches={self.batch_count} "
            f"batch_size={last_batch_items} "
            f"max_len={last_batch_max_seq_len} "
            f"q={queue_size}/{queue_max} "
            f"tokenize~{_fmt_ms(tok)} "
            f"forward~{_fmt_ms(fwd)} "
            f"index~{_fmt_ms(idx)} "
            f"cpu_copy~{_fmt_ms(cpy)} "
            f"put_block~{_fmt_ms(put)} "
            f"empty_cache~{_fmt_ms(ec)} "
            f"batch_total~{_fmt_ms(tot)}"
        )


def load_rollout_file(rollout_path: Path) -> Dict[str, Any]:
    """
    Load a rollout JSON file.
    
    Args:
        rollout_path: Path to rollout JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(rollout_path) as f:
        return json.load(f)


def find_sentence_position_in_text(full_text: str, sentence: str, sentence_idx: int) -> int:
    """
    Find the end position of a sentence in the full rollout text.
    
    Args:
        full_text: The full rollout text
        sentence: The sentence text to find
        sentence_idx: Sentence index (for debugging)
        
    Returns:
        Character position where the sentence ends in full_text
    """
    # Find the sentence in the text
    pos = full_text.find(sentence)
    if pos == -1:
        # Try matching just the beginning
        sentence_start = sentence[:50] if len(sentence) > 50 else sentence
        pos = full_text.find(sentence_start)
        if pos == -1:
            debug_print(f"  WARNING: Could not find sentence {sentence_idx} in full_text")
            return -1
    
    return pos + len(sentence)


def reconstruct_full_text(
    formatted_prompt: List[Dict[str, str]],
    thinking_prefix: str,
    counterfactual: str,
    tokenizer: Any
) -> str:
    """
    Reconstruct the full formatted text for a counterfactual continuation.
    
    Args:
        formatted_prompt: Original prompt messages (may include system turns)
        thinking_prefix: Full rollout text up to and including the branch sentence
        counterfactual: The counterfactual continuation from that point
        tokenizer: Tokenizer for chat template
        
    Returns:
        Fully formatted text with chat template applied
    """
    # Full assistant response = thinking_prefix + counterfactual
    assistant_content = thinking_prefix + counterfactual
    
    # Create messages (reuse original messages to preserve formatting)
    messages = list(formatted_prompt)
    messages.append({"role": "assistant", "content": assistant_content})
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return formatted


def find_assistant_range(
    formatted_text: str,
    token_ids: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    assistant_content: str,
    tokenizer: Any
) -> Tuple[int, int]:
    """
    Find the token range corresponding to assistant_content inside formatted_text,
    using offset mappings (robust vs re-tokenizing prefixes).
    """
    assistant_start_char = formatted_text.find(assistant_content)
    if assistant_start_char == -1:
        # Fallback: try anchoring on the beginning of assistant_content
        anchor = assistant_content[:200] if len(assistant_content) > 200 else assistant_content
        if anchor:
            assistant_start_char = formatted_text.find(anchor)
    if assistant_start_char == -1:
        # Last resort: approximate to the second half of the sequence.
        seq_len = len(token_ids)
        return seq_len // 2, max(seq_len - 2, 0)
    
    assistant_end_char = assistant_start_char + len(assistant_content)
    
    # Convert char positions -> token indices
    start_idx = 0
    end_idx = len(token_ids) - 1
    
    # Start token
    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start <= assistant_start_char < char_end:
            start_idx = tok_idx
            break
        if char_start >= assistant_start_char:
            start_idx = tok_idx
            break
    
    # End token (search backward)
    for tok_idx in range(len(offset_mapping) - 1, -1, -1):
        char_start, char_end = offset_mapping[tok_idx]
        if char_start < assistant_end_char <= char_end:
            end_idx = tok_idx
            break
        if char_end <= assistant_end_char:
            end_idx = tok_idx
            break
    
    # Trim special tokens at end
    special_ids = set(tokenizer.all_special_ids)
    while end_idx > start_idx and token_ids[end_idx].item() in special_ids:
        end_idx -= 1
    
    if start_idx >= end_idx:
        start_idx = max(0, end_idx - 32)
    
    return start_idx, end_idx


def select_random_positions(
    start_idx: int,
    end_idx: int,
    num_positions: int,
    token_ids: torch.Tensor,
    tokenizer: Any,
    seed: int
) -> List[int]:
    """
    Select random token positions within the range, avoiding special tokens.
    """
    random.seed(seed)
    
    special_ids = set(tokenizer.all_special_ids)
    
    # Get all valid (non-special) positions in range
    valid_positions = []
    for pos in range(start_idx, min(end_idx + 1, len(token_ids))):
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
    """
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
    
    # Remove batch dimension if present
    if all_activations.dim() == 4:
        all_activations = all_activations.squeeze(1)
    
    # Extract at specified positions: (num_layers, num_positions, hidden_dim)
    position_activations = all_activations[:, positions, :]
    del all_activations
    
    return position_activations


def extract_activations_batched(
    formatted_texts: List[str],
    model: Any,
    layer_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Run ONE batched forward pass with nnsight and return the full activation tensor.
    
    Returns:
        Tensor of shape:
          - (num_layers, batch, seq_len, hidden_dim) for batch > 1 (typical)
          - (num_layers, seq_len, hidden_dim) for batch == 1 (after squeezing)
    """
    saved_outputs = []
    with torch.no_grad():
        # nnsight supports batching by passing a list of prompts
        with model.trace(formatted_texts):
            if layer_idx is not None:
                num_layers_model = len(model.model.layers)
                if layer_idx >= num_layers_model or layer_idx < 0:
                    raise ValueError(f"layer_idx {layer_idx} out of range [0, {num_layers_model})")
                saved_outputs.append(model.model.layers[layer_idx].output[0].save())
            else:
                for layer in model.model.layers:
                    saved_outputs.append(layer.output[0].save())

    # IMPORTANT:
    # Stack on CPU to reduce GPU peak memory. When VRAM is tight (e.g., model already
    # occupies most of an 80GB card), stacking all layer outputs on GPU can push us
    # over the edge and can also "poison" subsequent steps after an OOM.
    all_activations = torch.stack([out.cpu() for out in saved_outputs], dim=0)
    del saved_outputs
    
    # Handle shape variants:
    # - sometimes there's a singleton dimension for batch==1
    if all_activations.dim() == 4 and all_activations.shape[1] == 1:
        all_activations = all_activations.squeeze(1)  # (num_layers, seq_len, hidden)
    
    return all_activations


def save_sample_activations(
    sample_id: str,
    token_ids: torch.Tensor,
    positions: List[int],
    activations: torch.Tensor,
    label: float,
    metadata: Dict[str, Any],
    output_dir: str,
    tokenizer: Any
) -> None:
    """
    Save activations for a single sample in format compatible with quick_load.
    
    Key difference from counterfactual: ALL positions get the SAME label (smeared).
    """
    # Create directory
    sample_dir = Path(output_dir) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Build sentence_data array for metadata.json
    sentence_data = []
    
    for pos_idx, token_pos in enumerate(positions):
        # Extract activation for this position: (num_layers, 1, hidden_dim)
        pos_activation = activations[:, pos_idx:pos_idx+1, :]
        
        # Get token text for this position
        token_text = tokenizer.decode([token_ids[token_pos].item()])
        
        # Save as sentence_{pos_idx}.pt in format (token_ids, activations)
        pt_path = sample_dir / f"sentence_{pos_idx}.pt"
        torch.save((token_ids.cpu(), pos_activation.cpu()), pt_path)
        
        # Add to sentence_data array - SAME label for all (smeared)
        sentence_data.append({
            "sentence_index": pos_idx,
            "sentence": token_text,
            "p_reward_hacks": label,  # SMEARED: same for all positions
            "token_position": token_pos,
        })
    
    # Create metadata.json
    metadata_json = {
        "sentence_data": sentence_data,
        "sample_type": "baseline",
        "labeling_method": "smeared_binary",
        "source_file": metadata.get("source_file"),
        "sentence_idx": metadata.get("sentence_idx"),
        "rollout_idx": metadata.get("rollout_idx"),
        "is_hacking": metadata.get("is_hacking"),
        "num_layers": activations.shape[0],
        "hidden_dim": activations.shape[2],
    }
    
    metadata_path = sample_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_json, f, indent=2)


def _save_worker(
    save_q: "queue.Queue[Optional[Dict[str, Any]]]",
    tokenizer: Any,
    *,
    perf_log: bool = False,
    perf_log_every: int = 200,
    perf_save_warn_s: float = PERF_SAVE_WARN_S,
) -> None:
    """
    Background worker to write samples to disk.
    
    Expects items with keys:
      - sample_id, token_ids (cpu Tensor), positions (List[int]),
        activations (cpu Tensor), label (float), metadata (dict), token_texts (List[str])
    """
    n = 0
    ema_item_s = _PerfEMA()
    ema_pt_save_s = _PerfEMA()
    ema_json_save_s = _PerfEMA()
    while True:
        item = save_q.get()
        if item is None:
            save_q.task_done()
            break
        try:
            t_item0 = time.perf_counter()
            # Write exactly the same on-disk format as save_sample_activations()
            sample_id = item["sample_id"]
            token_ids = item["token_ids"]
            positions = item["positions"]
            activations = item["activations"]
            label = float(item["label"])
            metadata = item["metadata"]
            output_dir = item["output_dir"]
            token_texts = item["token_texts"]
            
            sample_dir = Path(output_dir) / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            sentence_data = []
            t_pt_total = 0.0
            for pos_idx, token_pos in enumerate(positions):
                pos_activation = activations[:, pos_idx:pos_idx+1, :]
                token_text = token_texts[pos_idx]
                
                pt_path = sample_dir / f"sentence_{pos_idx}.pt"
                t0 = time.perf_counter()
                torch.save((token_ids, pos_activation), pt_path)
                t_pt_total += time.perf_counter() - t0
                
                sentence_data.append({
                    "sentence_index": pos_idx,
                    "sentence": token_text,
                    "p_reward_hacks": label,
                    "token_position": int(token_pos),
                })
            
            metadata_json = {
                "sentence_data": sentence_data,
                "sample_type": "baseline",
                "labeling_method": "smeared_binary",
                "source_file": metadata.get("source_file"),
                "sentence_idx": metadata.get("sentence_idx"),
                "rollout_idx": metadata.get("rollout_idx"),
                "is_hacking": metadata.get("is_hacking"),
                "num_layers": int(activations.shape[0]),
                "hidden_dim": int(activations.shape[2]),
            }
            metadata_path = sample_dir / "metadata.json"
            t0 = time.perf_counter()
            with open(metadata_path, "w") as f:
                json.dump(metadata_json, f, indent=2)
            t_json = time.perf_counter() - t0

            t_item = time.perf_counter() - t_item0
            n += 1
            ema_item_s.update(t_item)
            ema_pt_save_s.update(t_pt_total)
            ema_json_save_s.update(t_json)

            if perf_log:
                if t_item >= float(perf_save_warn_s):
                    print(
                        "[PERF][SAVE][SLOW] "
                        f"item={sample_id} took={_fmt_ms(t_item)} "
                        f"(pt_total={_fmt_ms(t_pt_total)}, json={_fmt_ms(t_json)}) "
                        f"q={save_q.qsize()}/{getattr(save_q, 'maxsize', -1)}"
                    )
                if perf_log_every > 0 and (n % perf_log_every == 0):
                    print(
                        "[PERF][SAVE] "
                        f"items={n} "
                        f"item~{_fmt_ms(ema_item_s.value or 0.0)} "
                        f"pt_total~{_fmt_ms(ema_pt_save_s.value or 0.0)} "
                        f"json~{_fmt_ms(ema_json_save_s.value or 0.0)} "
                        f"q={save_q.qsize()}/{getattr(save_q, 'maxsize', -1)}"
                    )
        finally:
            save_q.task_done()


def main():
    """Main extraction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract baseline activations with smeared binary labels")
    parser.add_argument("--rollouts-dir", type=str, default=str(ROLLOUTS_DIR),
                       help="Directory containing rollout JSON files")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="Output directory for activations")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                       help="Maximum number of samples to extract")
    parser.add_argument("--rollouts-per-sentence", type=int, default=ROLLOUTS_PER_SENTENCE,
                       help="Number of rollouts to use per sentence")
    parser.add_argument("--tokens-per-sample", type=int, default=TOKENS_PER_SAMPLE,
                       help="Random token positions per sample")
    parser.add_argument("--shuffle-files", action="store_true", default=True,
                       help="Shuffle rollout files before processing (default: True)")
    parser.add_argument("--no-shuffle-files", action="store_true",
                       help="Disable shuffling of rollout files")
    parser.add_argument("--shuffle-sentences", action="store_true", default=True,
                       help="Shuffle sentences within each rollout before sampling (default: True)")
    parser.add_argument("--no-shuffle-sentences", action="store_true",
                       help="Disable shuffling of sentences")
    parser.add_argument("--sample-rollouts-randomly", action="store_true", default=True,
                       help="Randomly choose which counterfactual rollouts to use per sentence (default: True)")
    parser.add_argument("--use-first-rollouts", action="store_true",
                       help="Use the first N rollouts per sentence instead of random sampling")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for nnsight forward passes (default: 8)")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                       help="Skip samples whose tokenized length exceeds this (default: 8192)")
    parser.add_argument("--long-seq-threshold", type=int, default=3072,
                       help="If seq_len exceeds this, use --long-seq-batch-size (default: 3072)")
    parser.add_argument("--long-seq-batch-size", type=int, default=1,
                       help="Batch size to use when encountering long sequences (default: 1)")
    parser.add_argument("--save-queue-size", type=int, default=128,
                       help="Max queued samples awaiting disk write (default: 128)")
    parser.add_argument("--perf-log", action="store_true",
                       help="Print perf timings/queue backpressure to diagnose stalls (default: False)")
    parser.add_argument("--perf-log-every", type=int, default=PERF_LOG_EVERY,
                       help="Print perf summary every N batches (default: 50)")
    parser.add_argument("--perf-put-warn-s", type=float, default=PERF_PUT_WARN_S,
                       help="Warn if save_q.put blocks longer than this (seconds)")
    parser.add_argument("--perf-save-warn-s", type=float, default=PERF_SAVE_WARN_S,
                       help="Warn if saver takes longer than this per item (seconds)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug
    
    print("=" * 70)
    print("BASELINE ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Rollouts dir: {args.rollouts_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Rollouts per sentence: {args.rollouts_per_sentence}")
    print(f"  Tokens per sample: {args.tokens_per_sample}")
    print(f"  Shuffle files: {args.shuffle_files and not args.no_shuffle_files}")
    print(f"  Shuffle sentences: {args.shuffle_sentences and not args.no_shuffle_sentences}")
    print(f"  Sample rollouts randomly: {args.sample_rollouts_randomly and not args.use_first_rollouts}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Long seq threshold: {args.long_seq_threshold}")
    print(f"  Long seq batch size: {args.long_seq_batch_size}")
    print(f"  Save queue size: {args.save_queue_size}")
    print(f"  Perf log: {args.perf_log}")
    if args.perf_log:
        print(f"  Perf log every: {args.perf_log_every} batches")
        print(f"  Perf put warn: {args.perf_put_warn_s}s")
        print(f"  Perf save warn: {args.perf_save_warn_s}s")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layer idx: {LAYER_IDX or 'all'}")
    print(f"  Seed: {args.seed}")
    print(f"  Debug mode: {DEBUG}")
    print()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # =========================================================================
    # Step 1: Scan rollout files
    # =========================================================================
    print("=" * 70)
    print("SCANNING ROLLOUT FILES")
    print("=" * 70)
    
    rollouts_path = Path(args.rollouts_dir)
    rollout_files = sorted(rollouts_path.glob("*_rollouts.json"))
    rollout_files = [f for f in rollout_files if not f.name.startswith('_')]
    
    # Shuffle files to avoid early max-samples being dominated by the first prompt/files.
    if args.shuffle_files and not args.no_shuffle_files:
        random.shuffle(rollout_files)
    
    print(f"  Found {len(rollout_files)} rollout files")
    
    if len(rollout_files) == 0:
        print("ERROR: No rollout files found!")
        return 1
    
    # =========================================================================
    # Step 2: Initialize model and tokenizer
    # =========================================================================
    print("\n" + "=" * 70)
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
    
    try:
        num_layers = len(model.model.layers)
        hidden_size = model.config.hidden_size
        print(f"  Model loaded: {num_layers} layers, hidden_size={hidden_size}")
    except Exception as e:
        print(f"  Model loaded (could not get layer info: {e})")
    
    # =========================================================================
    # Step 3: Process rollouts and extract activations
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXTRACTING ACTIVATIONS")
    print("=" * 70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    total_samples = 0
    total_hacking = 0
    total_clean = 0
    successful = 0
    failed = 0
    
    sample_seed = args.seed
    
    # Async saver queue/thread
    save_q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=max(1, args.save_queue_size))
    saver = threading.Thread(
        target=_save_worker,
        args=(save_q, tokenizer),
        kwargs={
            "perf_log": bool(args.perf_log),
            "perf_log_every": 200,
            "perf_save_warn_s": float(args.perf_save_warn_s),
        },
        daemon=True,
    )
    saver.start()
    
    # Batch accumulation
    batch_items: List[Dict[str, Any]] = []
    perf = _PerfCounters()
    t_tokenize_accum_s = 0.0

    def _flush_batch() -> None:
        """Run one nnsight forward pass for the current batch and enqueue saves."""
        nonlocal batch_items
        if not batch_items:
            return

        t_batch0 = time.perf_counter()
        texts = [b["formatted_text"] for b in batch_items]
        batch_max_len = max(int(b.get("seq_len", 0)) for b in batch_items)
        try:
            t0 = time.perf_counter()
            all_acts = extract_activations_batched(texts, model, LAYER_IDX)
            t_forward = time.perf_counter() - t0
        except Exception as e:
            # Robust OOM recovery: if we OOM mid-trace, we want to drop the current
            # batch and aggressively release cached memory so we don't "spiral".
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
            if is_oom:
                print(
                    "[WARN] CUDA OOM during batched forward "
                    f"(batch_items={len(batch_items)}, max_len={batch_max_len}). "
                    "Dropping this batch and clearing CUDA cache."
                )
                batch_items.clear()
                try:
                    gc.collect()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return
            raise

        # all_acts shape:
        # - (num_layers, batch, seq, hidden) OR (num_layers, seq, hidden) if batch==1
        t_index_total = 0.0
        t_cpu_total = 0.0
        t_put_total = 0.0
        for bi, b in enumerate(batch_items):
            pos = b["positions"]
            t0 = time.perf_counter()
            if all_acts.dim() == 4:
                acts = all_acts[:, bi, pos, :]
            else:
                acts = all_acts[:, pos, :]
            t_index_total += time.perf_counter() - t0

            # Enqueue CPU tensors for async writer
            t0 = time.perf_counter()
            acts_cpu = acts.cpu()
            t_cpu_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            save_q.put({
                "sample_id": b["sample_id"],
                "token_ids": b["token_ids"].cpu(),
                "positions": b["positions"],
                "activations": acts_cpu,
                "label": b["label"],
                "metadata": b["metadata"],
                "output_dir": args.output_dir,
                "token_texts": b["token_texts"],
            })
            t_put = time.perf_counter() - t0
            t_put_total += t_put
            if args.perf_log and t_put >= float(args.perf_put_warn_s):
                print(
                    "[PERF][PUT][BLOCKED] "
                    f"blocked={_fmt_ms(t_put)} "
                    f"q={save_q.qsize()}/{getattr(save_q, 'maxsize', -1)}"
                )

        batch_items.clear()
        del all_acts

        # Mild cleanup to reduce fragmentation
        if torch.cuda.is_available():
            t0 = time.perf_counter()
            torch.cuda.empty_cache()
            t_empty_cache = time.perf_counter() - t0
        else:
            t_empty_cache = 0.0

        # Update perf EMAs and print summary periodically
        perf.batch_count += 1
        perf.ema_tokenize_s.update(t_tokenize_accum_s)
        perf.ema_forward_s.update(t_forward)
        perf.ema_index_s.update(t_index_total)
        perf.ema_cpu_copy_s.update(t_cpu_total)
        perf.ema_put_block_s.update(t_put_total)
        perf.ema_empty_cache_s.update(t_empty_cache)
        perf.ema_batch_total_s.update(time.perf_counter() - t_batch0)
        if args.perf_log:
            perf.maybe_print(
                every=int(args.perf_log_every),
                queue_size=save_q.qsize(),
                queue_max=int(getattr(save_q, "maxsize", -1)),
                last_batch_items=len(texts),
                last_batch_max_seq_len=int(batch_max_len),
            )
    
    for file_idx, rollout_file in enumerate(rollout_files):
        if total_samples >= args.max_samples:
            print(f"\n  Reached max samples limit ({args.max_samples})")
            break
        
        print(f"\n  [{file_idx+1}/{len(rollout_files)}] Processing: {rollout_file.name}")
        
        try:
            # Load rollout data
            data = load_rollout_file(rollout_file)
            full_rollout_text = data.get('full_rollout_text', '')
            sentence_data = data.get('sentence_data', [])
            
            if not sentence_data:
                debug_print(f"    Skipping: no sentence_data")
                continue
            
            debug_print(f"    {len(sentence_data)} sentences, {len(full_rollout_text)} chars")
            
            # Get user prompt from first sentence
            first_sentence = sentence_data[0]
            formatted_prompt = first_sentence.get('formatted_prompt', [])
            if not formatted_prompt:
                debug_print(f"    Skipping: no formatted_prompt")
                continue
            
            # Sort by sentence_index so we can compute sentence end-positions robustly.
            sentence_data_sorted = sorted(sentence_data, key=lambda s: s.get("sentence_index", 0))
            
            # Precompute the end character position for each sentence using sequential search,
            # then we can shuffle sentence order without breaking prefix reconstruction.
            end_pos_by_sentence_idx: Dict[int, int] = {}
            cumulative_pos = 0
            for s in sentence_data_sorted:
                idx = int(s.get("sentence_index", 0))
                txt = s.get("sentence", "")
                if not txt:
                    continue
                sent_pos = full_rollout_text.find(txt, cumulative_pos)
                if sent_pos == -1:
                    # Fallback: try from beginning (less safe if duplicates exist)
                    sent_pos = full_rollout_text.find(txt)
                if sent_pos == -1:
                    continue
                sentence_end = sent_pos + len(txt)
                end_pos_by_sentence_idx[idx] = sentence_end
                cumulative_pos = sentence_end
            
            # Choose iteration order over sentences
            sentence_data_iter = list(sentence_data_sorted)
            if args.shuffle_sentences and not args.no_shuffle_sentences:
                random.shuffle(sentence_data_iter)
            
            for sent in sentence_data_iter:
                if total_samples >= args.max_samples:
                    break
                
                sentence_idx = int(sent.get("sentence_index", 0))
                sentence_text = sent.get('sentence', '')
                rollouts = sent.get('rollouts', [])
                
                if not rollouts:
                    debug_print(f"    Sentence {sentence_idx}: no rollouts")
                    continue
                
                sentence_end = end_pos_by_sentence_idx.get(sentence_idx, -1)
                if sentence_end == -1:
                    debug_print(f"    Sentence {sentence_idx}: could not locate end position in text")
                    continue
                
                # Get thinking prefix (everything up to and including this sentence)
                thinking_prefix = full_rollout_text[:sentence_end]
                
                # Process first N rollouts for this sentence
                num_rollouts_to_use = min(args.rollouts_per_sentence, len(rollouts))
                if num_rollouts_to_use <= 0:
                    continue
                
                if args.sample_rollouts_randomly and not args.use_first_rollouts:
                    chosen_rollout_indices = random.sample(range(len(rollouts)), k=num_rollouts_to_use)
                    chosen_rollout_indices.sort()
                else:
                    chosen_rollout_indices = list(range(num_rollouts_to_use))
                
                for rollout_idx in chosen_rollout_indices:
                    if total_samples >= args.max_samples:
                        break
                    
                    counterfactual = rollouts[rollout_idx]
                    
                    # Determine binary label
                    is_hacking = detect_reward_hacking(counterfactual, ignore_reasoning=True)
                    label = 1.0 if is_hacking else 0.0
                    
                    if is_hacking:
                        total_hacking += 1
                    else:
                        total_clean += 1
                    
                    sample_id = f"baseline_{rollout_file.stem}_s{sentence_idx}_r{rollout_idx}"
                    
                    debug_print(f"    Sample {sample_id}: hacking={is_hacking}")
                    
                    try:
                        # Reconstruct full formatted text
                        formatted_text = reconstruct_full_text(
                            formatted_prompt=formatted_prompt,
                            thinking_prefix=thinking_prefix,
                            counterfactual=counterfactual,
                            tokenizer=tokenizer,
                        )
                        
                        if DEBUG and total_samples < 3:
                            print(f"\n    [DEBUG] Sample {total_samples}:")
                            print(f"      Thinking prefix: {len(thinking_prefix)} chars")
                            print(f"      Counterfactual: {len(counterfactual)} chars")
                            print(f"      Formatted: {len(formatted_text)} chars")
                            print(f"      Label: {label} (hacking={is_hacking})")
                            print(f"      First 200 chars: {formatted_text[:200]}...")
                            print(f"      Last 200 chars: ...{formatted_text[-200:]}")
                        
                        assistant_content = thinking_prefix + counterfactual
                        
                        # Tokenize with offset mapping (robust span localization)
                        t0 = time.perf_counter()
                        encoding = tokenizer(
                            formatted_text,
                            return_offsets_mapping=True,
                            return_tensors="pt",
                        )
                        t_tokenize_accum_s += time.perf_counter() - t0
                        token_ids = encoding["input_ids"][0]
                        offset_mapping = encoding["offset_mapping"][0].tolist()
                        seq_len = int(token_ids.shape[0])
                        if seq_len > int(args.max_seq_len):
                            debug_print(
                                f"      Skipping: seq_len={seq_len} exceeds max_seq_len={args.max_seq_len}"
                            )
                            failed += 1
                            continue
                        
                        # Find assistant response range
                        start_idx, end_idx = find_assistant_range(
                            formatted_text=formatted_text,
                            token_ids=token_ids,
                            offset_mapping=offset_mapping,
                            assistant_content=assistant_content,
                            tokenizer=tokenizer,
                        )
                        
                        debug_print(f"      Token range: [{start_idx}, {end_idx}] of {len(token_ids)}")
                        
                        # Select random positions
                        sample_seed += 1
                        positions = select_random_positions(
                            start_idx, end_idx, args.tokens_per_sample,
                            token_ids, tokenizer, sample_seed
                        )
                        
                        if not positions:
                            debug_print(f"      Skipping: no valid positions")
                            failed += 1
                            continue
                        
                        debug_print(f"      Positions: {positions}")
                        
                        # Stash for batched forward pass
                        token_texts = [tokenizer.decode([token_ids[p].item()]) for p in positions]
                        item_is_long = seq_len > int(args.long_seq_threshold)
                        if batch_items:
                            batch_is_long = bool(batch_items[0].get("is_long", False))
                            if item_is_long != batch_is_long:
                                _flush_batch()
                                t_tokenize_accum_s = 0.0

                        batch_items.append({
                            "sample_id": sample_id,
                            "formatted_text": formatted_text,
                            "token_ids": token_ids,  # keep on CPU for now; also used for decoding/positions
                            "positions": positions,
                            "label": label,
                            "metadata": {
                                "source_file": rollout_file.name,
                                "sentence_idx": sentence_idx,
                                "rollout_idx": rollout_idx,
                                "is_hacking": is_hacking,
                            },
                            "token_texts": token_texts,
                            "seq_len": seq_len,
                            "is_long": item_is_long,
                        })
                        
                        # If batch is full, run one nnsight forward pass and enqueue saves
                        effective_batch_size = int(args.long_seq_batch_size) if item_is_long else int(args.batch_size)
                        effective_batch_size = max(1, effective_batch_size)
                        if len(batch_items) >= effective_batch_size:
                            _flush_batch()
                            t_tokenize_accum_s = 0.0
                        
                        total_samples += 1
                        successful += 1
                        
                        if total_samples % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = total_samples / elapsed
                            print(f"    Progress: {total_samples}/{args.max_samples} samples "
                                  f"({rate:.1f}/s, {100*total_hacking/total_samples:.1f}% hacking)")
                        
                        # Periodic cleanup (doing this every sample hurts throughput)
                        if total_samples % 250 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                    except Exception as e:
                        debug_print(f"      Error: {e}")
                        failed += 1
                        continue
                
        except Exception as e:
            print(f"    Error processing file: {e}")
            continue
    
    # =========================================================================
    # Summary
    # =========================================================================
    # Flush any remaining batch
    if batch_items:
        _flush_batch()
    
    # Wait for all writes to finish
    save_q.put(None)
    save_q.join()
    saver.join(timeout=5)

    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Hacking samples: {total_hacking} ({100*total_hacking/(total_samples or 1):.1f}%)")
    print(f"  Clean samples: {total_clean} ({100*total_clean/(total_samples or 1):.1f}%)")
    print(f"  Time: {elapsed:.1f}s ({total_samples/elapsed:.2f} samples/s)")
    print(f"  Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

