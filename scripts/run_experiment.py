#!/usr/bin/env python3
"""
Main script to run the counterfactual probing data collection experiment.

Configure parameters at the top of this file, then run:
    python run_experiment.py
"""

import sys
from pathlib import Path
import time
import gc
import torch

# Add src to path
# scripts/ is a sibling of src/, so we need parent.parent here.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Progress bars will be limited without tqdm.\n")

try:
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("Error: vllm not installed. Install with: pip install vllm")
    sys.exit(1)

from rollouts.generate import RolloutGenerator


# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Data files
PROMPTS_FILE = "src/data/reward_hacking_prompts.json"
OUTPUT_DIR = "src/rollouts"

# Model configuration
MODEL = "openai/gpt-oss-20b"
TENSOR_PARALLEL_SIZE = 1  # Number of GPUs to use for tensor parallelism
GPU_MEMORY_UTILIZATION = 0.9  # Fraction of GPU memory to use (reduced to leave headroom for better batching)

# Generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = 8000  # Reduced from 32000 for better memory efficiency (most completions don't need 32K)

# Experiment parameters
NUM_COUNTERFACTUALS = 50  # Number of rollouts per sentence
NUM_INITIAL_ROLLOUTS = 5 # Number of initial rollouts per prompt (typically 1)
N_INITIAL_ROLLOUT_TESTS = 1  # Number of initial rollouts to generate and test before selecting one
INITIAL_ROLLOUT_SELECTION_STRATEGY = "MEDIAN"  # Strategy for selecting from N_INITIAL_ROLLOUT_TESTS: "MIN" or "MEDIAN"
NUM_SENTENCES_TO_SAMPLE = 20  # Number of evenly-spaced sentences to process (None = process all sentences)

# Batching configuration
COUNTERFACTUAL_BATCH_SIZE = 3000  # Process counterfactuals in batches to avoid memory pressure

# Parallelization (no longer used - vLLM handles batching internally)
# MAX_WORKERS removed - we use batched generation instead

# Processing options
PROCESS_ALL_PROMPTS = True  # If False, will process only the first prompt
PROMPT_INDICES = None  # Optional: specify which prompts to process (e.g., [0, 1, 2])
                       # Only used if PROCESS_ALL_PROMPTS is False
SKIP_IF_EXISTS = True  # Skip prompts that already have output files (for resuming)

# ============================================================================


def main():
    """Run the counterfactual probing experiment."""
    
    print("=" * 70)
    print("COUNTERFACTUAL PROBING DATA COLLECTION EXPERIMENT")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Prompts file: {PROMPTS_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Model: {MODEL}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Counterfactuals per sentence: {NUM_COUNTERFACTUALS}")
    print(f"  Initial rollouts per prompt: {NUM_INITIAL_ROLLOUTS}")
    print(f"  Initial rollout tests: {N_INITIAL_ROLLOUT_TESTS}")
    print(f"  Initial rollout selection: {INITIAL_ROLLOUT_SELECTION_STRATEGY}")
    print(f"  Sentences to sample: {NUM_SENTENCES_TO_SAMPLE if NUM_SENTENCES_TO_SAMPLE is not None else 'all'}")
    print(f"  Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    print(f"  GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    print(f"  Skip if exists: {SKIP_IF_EXISTS}")
    print()
    
    # Initialize vLLM engine ONCE at startup
    print("Initializing vLLM engine...")
    print("  This may take a minute to load the model...")
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_prefix_caching=True,  # Enable prefix caching for shared prompt prefixes
        max_num_seqs=1024,  # Increase from default 256 - allows more concurrent sequences
        max_num_batched_tokens=32768,  # Increase from default 8192 - allows larger batches
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    # For counterfactual generation - use same params as initial rollouts
    # (We duplicate prompts instead of using n= parameter for better batching)
    counterfactual_sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    print("  ✓ vLLM engine initialized\n")
    
    # Cache tokenizer once
    print("Caching tokenizer...")
    tokenizer = None
    try:
        tokenizer = getattr(llm, 'llm_engine', None)
        if tokenizer:
            tokenizer = getattr(tokenizer, 'tokenizer', None)
    except:
        pass
    if tokenizer:
        print("  ✓ Tokenizer cached\n")
    else:
        print("  ⚠ Tokenizer not found, will use fallback formatting\n")
    
    # Initialize generator (no longer needs API URL or workers)
    generator = RolloutGenerator(
        model=MODEL,
        n_initial_rollout_tests=N_INITIAL_ROLLOUT_TESTS,
        initial_rollout_selection_strategy=INITIAL_ROLLOUT_SELECTION_STRATEGY,
        num_sentences_to_sample=NUM_SENTENCES_TO_SAMPLE
    )
    generator.temperature = TEMPERATURE
    generator.max_tokens = MAX_TOKENS
    generator.num_counterfactuals = NUM_COUNTERFACTUALS
    generator.llm = llm  # Pass the LLM engine
    generator.sampling_params = sampling_params  # Pass sampling params for initial rollouts
    generator.counterfactual_sampling_params = counterfactual_sampling_params  # Pass sampling params for counterfactuals
    generator.tokenizer = tokenizer  # Cache tokenizer
    
    # Load prompts
    import json
    prompts_path = Path(PROMPTS_FILE)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {PROMPTS_FILE}")
        return 1
    
    with open(prompts_path) as f:
        all_prompts = json.load(f)
    
    print(f"Loaded {len(all_prompts)} prompt(s) from {PROMPTS_FILE}")
    
    # Determine which prompts to process
    if PROCESS_ALL_PROMPTS:
        prompts_to_process = all_prompts
    elif PROMPT_INDICES is not None:
        prompts_to_process = [all_prompts[i] for i in PROMPT_INDICES if 0 <= i < len(all_prompts)]
    else:
        prompts_to_process = all_prompts[:1]  # Default to first prompt
    
    print(f"Processing {len(prompts_to_process)} prompt(s)\n")
    print("=" * 70)
    
    experiment_start = time.time()
    results = []
    
    try:
        # Phase 1: Prepare and generate ALL initial rollouts
        if HAS_TQDM:
            print("\n" + "=" * 70)
        all_initial_prompts, initial_metadata, prompt_info_list = generator.prepare_all_initial_rollouts(
            prompts_to_process,
            OUTPUT_DIR,
            SKIP_IF_EXISTS,
            NUM_INITIAL_ROLLOUTS
        )
        
        if all_initial_prompts:
            if HAS_TQDM:
                pbar_gen = tqdm(total=len(all_initial_prompts), desc="Phase 1: Generating initial rollouts", unit="request")
            else:
                pbar_gen = None
                print(f"\nPhase 1: Generating {len(all_initial_prompts)} initial rollouts...")
            
            try:
                initial_outputs = generator.llm.generate(all_initial_prompts, generator.sampling_params)
            except Exception as e:
                print(f"\n✗ Error generating initial rollouts: {e}")
                import traceback
                traceback.print_exc()
                return 1
            
            if pbar_gen:
                pbar_gen.close()
            if not HAS_TQDM:
                print(f"  ✓ Generated {len(initial_outputs)} initial rollouts")
        else:
            initial_outputs = []
            if not HAS_TQDM:
                print("\nNo initial rollouts to generate (all prompts already processed)")
        
        # Phase 2: Process and select best rollouts
        if HAS_TQDM:
            print("\n" + "=" * 70)
        try:
            processed_prompts = generator.process_initial_rollouts(
                initial_outputs,
                initial_metadata,
                prompt_info_list,
                NUM_INITIAL_ROLLOUTS
            )
        except Exception as e:
            print(f"\n✗ Error processing initial rollouts: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        if not processed_prompts:
            print("\nNo prompts to process after initial rollout selection")
            return 0
        
        # Save checkpoint of initial rollouts for recovery
        try:
            generator.save_initial_rollouts_checkpoint(processed_prompts, OUTPUT_DIR)
        except Exception as e:
            print(f"  ⚠ Warning: Could not save checkpoint: {e}")
        
        # Phase 3: Prepare ALL counterfactual prompts (with resume support)
        if HAS_TQDM:
            print("\n" + "=" * 70)
        all_counterfactual_prompts, counterfactual_metadata = generator.prepare_all_counterfactuals(
            processed_prompts,
            output_dir=OUTPUT_DIR,
            skip_completed=SKIP_IF_EXISTS
        )
        
        # Phase 4: Generate counterfactuals in prompt-aligned batches
        if all_counterfactual_prompts:
            total_prompts = len(all_counterfactual_prompts)
            
            # Find prompt boundaries for prompt-aligned batching
            batch_boundaries = [0]
            if counterfactual_metadata:
                current_prompt_idx = counterfactual_metadata[0][0]
                for i, (prompt_idx, _, _) in enumerate(counterfactual_metadata[1:], start=1):
                    if prompt_idx != current_prompt_idx:
                        batch_boundaries.append(i)
                        current_prompt_idx = prompt_idx
            batch_boundaries.append(total_prompts)
            
            num_batches = len(batch_boundaries) - 1
            
            if HAS_TQDM:
                pbar_cf = tqdm(total=total_prompts, desc="Phase 4: Generating counterfactuals", unit="request")
            else:
                pbar_cf = None
                print(f"\nPhase 4: Generating {total_prompts} counterfactuals in {num_batches} prompt-aligned batches...")
            
            all_counterfactual_outputs = []
            
            try:
                for batch_idx in range(num_batches):
                    start_idx = batch_boundaries[batch_idx]
                    end_idx = batch_boundaries[batch_idx + 1]
                    batch_prompts = all_counterfactual_prompts[start_idx:end_idx]
                    batch_metadata = counterfactual_metadata[start_idx:end_idx]
                    
                    # Get prompt info for logging
                    batch_prompt_idx = batch_metadata[0][0] if batch_metadata else None
                    batch_size = len(batch_prompts)
                    
                    if not HAS_TQDM:
                        print(f"  Batch {batch_idx + 1}/{num_batches}: Processing {batch_size} prompts (prompt {batch_prompt_idx})...")
                    
                    # Generate this batch
                    batch_outputs = generator.llm.generate(batch_prompts, generator.counterfactual_sampling_params)
                    all_counterfactual_outputs.extend(batch_outputs)
                    
                    # Cleanup between batches to reduce memory fragmentation
                    # This helps prevent slowdown in subsequent batches
                    del batch_outputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    if pbar_cf:
                        pbar_cf.update(batch_size)
                    
                    if not HAS_TQDM:
                        print(f"    ✓ Generated {batch_size} completions")
                
                counterfactual_outputs = all_counterfactual_outputs
                
            except Exception as e:
                print(f"\n✗ Error generating counterfactuals: {e}")
                import traceback
                traceback.print_exc()
                print("\n⚠ Partial results may have been saved. You can resume by running again.")
                return 1
            
            if pbar_cf:
                pbar_cf.close()
            if not HAS_TQDM:
                print(f"  ✓ Generated {len(counterfactual_outputs)} counterfactuals total")
        else:
            counterfactual_outputs = []
            if not HAS_TQDM:
                print("\nNo counterfactuals to generate (all already completed)")
        
        # Phase 5: Process and save all results (with progressive saving)
        if HAS_TQDM:
            print("\n" + "=" * 70)
        try:
            results_dict = generator.process_all_counterfactuals(
                counterfactual_outputs,
                counterfactual_metadata,
                processed_prompts,
                output_dir=OUTPUT_DIR,
                save_progressively=True
            )
        except Exception as e:
            print(f"\n✗ Error processing counterfactuals: {e}")
            import traceback
            traceback.print_exc()
            print("\n⚠ Some results may have been saved. You can resume by running again.")
            return 1
        
        # Phase 6: Verify all results are saved (progressive saving already did most of the work)
        results = generator.save_all_results(
            results_dict,
            processed_prompts,
            OUTPUT_DIR,
            force_resave=False
        )
        
    except Exception as e:
        print(f"\n✗ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    total_time = time.time() - experiment_start
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nProcessed {len(results)} prompt(s) successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    
    if results:
        total_sentences = sum(r["num_sentences"] for r in results)
        total_rollouts = total_sentences * NUM_COUNTERFACTUALS
        print(f"Total sentences processed: {total_sentences}")
        print(f"Total rollouts generated: {total_rollouts}")
        print(f"Average time per prompt: {total_time/len(results):.1f} seconds")
    
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print(f"\nOutput file names:")
    for result in results:
        prompt_name = result["prompt_name"]
        rollout_idx = result.get("rollout_index", None)
        if rollout_idx is not None:
            output_file = Path(OUTPUT_DIR) / f"{prompt_name}_rollout_{rollout_idx}_rollouts.json"
        else:
            output_file = Path(OUTPUT_DIR) / f"{prompt_name}_rollouts.json"
        if output_file.exists():
            file_size = output_file.stat().st_size / 1024  # KB
            print(f"  - {output_file} ({file_size:.1f} KB)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())