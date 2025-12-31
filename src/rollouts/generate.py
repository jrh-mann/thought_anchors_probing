"""
Main rollout generation script.
Orchestrates the full pipeline for counterfactual probing data collection.
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    # tqdm not available, use dummy function
    HAS_TQDM = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.chat_utils import concatenate_prompt, format_chat_messages, messages_to_prompt_string
from utils.reward_detector import detect_reward_hacking
from utils.text_utils import split_sentences, get_sentence_prefix
from rollouts.schema import create_sentence_data, validate_sentence_data


class RolloutGenerator:
    """Main class for generating rollouts and computing reward hacking probabilities."""
    
    def __init__(self, model: str = "openai/gpt-oss-20b", 
                 n_initial_rollout_tests: int = 10,
                 initial_rollout_selection_strategy: str = "MIN",
                 num_sentences_to_sample: Optional[int] = None):
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 32000
        self.num_counterfactuals = 50
        self.n_initial_rollout_tests = n_initial_rollout_tests  # Number of initial rollouts to test
        self.initial_rollout_selection_strategy = initial_rollout_selection_strategy.upper()  # MIN or MEDIAN
        self.num_sentences_to_sample = num_sentences_to_sample  # Number of evenly-spaced sentences to process (None = all)
        self.llm = None  # Will be set from run_experiment.py
        self.sampling_params = None  # Will be set from run_experiment.py (for initial rollouts)
        self.counterfactual_sampling_params = None  # Will be set from run_experiment.py (for counterfactuals with n=)
        self.tokenizer = None  # Will be set from run_experiment.py (cached tokenizer)
    
    def load_prompts(self, prompts_path: str) -> List[Dict[str, str]]:
        """Load reward hacking prompts from JSON file."""
        with open(prompts_path, 'r') as f:
            return json.load(f)
    
    def _get_output_path(self, prompt_name: str, rollout_idx: int, output_dir: str) -> Path:
        """Get the output path for a prompt/rollout combination."""
        return Path(output_dir) / f"{prompt_name}_rollout_{rollout_idx}_rollouts.json"
    
    def _save_result_file(self, output_path: Path, sentence_data_list: List[Dict[str, Any]], 
                          full_rollout_text: Optional[str] = None) -> bool:
        """
        Save a result file atomically (write to temp file, then rename).
        
        Args:
            output_path: Path to save the file
            sentence_data_list: List of sentence data dictionaries
            full_rollout_text: Optional full rollout text for reconstruction (new format)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Write to temp file first, then rename atomically
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                if full_rollout_text is not None:
                    # New format: dict with full_rollout_text and sentence_data
                    output_data = {
                        "full_rollout_text": full_rollout_text,
                        "sentence_data": sentence_data_list
                    }
                else:
                    # Old format: just list (backward compatibility)
                    output_data = sentence_data_list
                json.dump(output_data, f, indent=2)
            temp_path.replace(output_path)
            return True
        except Exception as e:
            print(f"  ⚠ Error saving {output_path}: {e}")
            # Clean up temp file if it exists
            temp_path = output_path.with_suffix('.tmp')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass
            return False
    
    def _load_existing_result(self, output_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load an existing result file if it exists and is valid.
        
        Returns:
            Dict with keys:
                - 'sentence_data': List of sentence data dictionaries
                - 'full_rollout_text': Optional full rollout text (None for old format)
            Or None if file doesn't exist or is invalid
        """
        if not output_path.exists():
            return None
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # Handle new format: dict with full_rollout_text and sentence_data
            if isinstance(data, dict):
                if "sentence_data" in data:
                    return {
                        "sentence_data": data["sentence_data"],
                        "full_rollout_text": data.get("full_rollout_text")
                    }
            
            # Handle old format: list of sentence data (backward compatibility)
            if isinstance(data, list) and len(data) > 0:
                # Quick validation: check if first item has required fields
                if isinstance(data[0], dict) and "sentence_index" in data[0]:
                    return {
                        "sentence_data": data,
                        "full_rollout_text": None  # Old format doesn't have this
                    }
            
            return None
        except Exception as e:
            # File might be corrupted or incomplete
            print(f"  ⚠ Could not load {output_path}: {e}")
            return None
    
    def _get_completed_prompts_rollouts(self, processed_prompts: List[Dict[str, Any]], 
                                        output_dir: str) -> Set[Tuple[int, int]]:
        """
        Get set of (prompt_idx, rollout_idx) tuples that are already completed.
        
        Returns:
            Set of (prompt_idx, rollout_idx) tuples
        """
        completed = set()
        prompt_idx_to_name = {p["prompt_idx"]: p["prompt_name"] for p in processed_prompts}
        
        for processed_prompt in processed_prompts:
            prompt_idx = processed_prompt["prompt_idx"]
            prompt_name = prompt_idx_to_name[prompt_idx]
            
            for rollout_data in processed_prompt["rollouts_data"]:
                rollout_idx = rollout_data["rollout_idx"]
                output_path = self._get_output_path(prompt_name, rollout_idx, output_dir)
                
                existing = self._load_existing_result(output_path)
                if existing is not None:
                    # Check if it has all expected sentences
                    expected_indices = set(rollout_data["selected_indices"])
                    existing_sentence_data = existing.get("sentence_data", [])
                    existing_indices = {s.get("sentence_index") for s in existing_sentence_data if isinstance(s, dict)}
                    
                    if expected_indices.issubset(existing_indices):
                        completed.add((prompt_idx, rollout_idx))
        
        return completed
    
    def prepare_all_initial_rollouts(
        self,
        prompts_to_process: List[Dict[str, str]],
        output_dir: str,
        skip_if_exists: bool,
        num_initial_rollouts: int
    ) -> tuple[List[str], List[tuple[int, int, int]], List[Dict[str, Any]]]:
        """
        Prepare all initial rollout prompts for all prompts.
        
        Args:
            prompts_to_process: List of prompt data dicts
            output_dir: Directory to check for existing outputs
            skip_if_exists: Whether to skip prompts that already have outputs
            num_initial_rollouts: Number of initial rollouts per prompt
        
        Returns:
            Tuple of (all_prompts_list, metadata_map, prompt_info_list)
            - all_prompts_list: List of prompt strings to generate
            - metadata_map: List of (prompt_idx, rollout_idx, test_idx) tuples mapping output index to prompt/rollout/test
            - prompt_info_list: List of dicts with prompt info (idx, name, needs_processing)
        """
        all_prompts = []
        metadata_map = []
        prompt_info_list = []
        
        if HAS_TQDM:
            pbar = tqdm(total=len(prompts_to_process), desc="Phase 1: Preparing initial rollout prompts", unit="prompt")
        else:
            pbar = None
            print("Phase 1: Preparing initial rollout prompts...")
        
        for prompt_idx, prompt_data in enumerate(prompts_to_process):
            prompt_name = prompt_data.get("name", f"prompt_{prompt_idx}")
            needs_processing = True
            
            # Check if all required rollouts already exist
            if skip_if_exists:
                all_exist = True
                for rollout_idx in range(num_initial_rollouts):
                    output_path = Path(output_dir) / f"{prompt_name}_rollout_{rollout_idx}_rollouts.json"
                    if not output_path.exists():
                        all_exist = False
                        break
                if all_exist:
                    needs_processing = False
            
            prompt_info_list.append({
                "prompt_idx": prompt_idx,
                "prompt_name": prompt_name,
                "prompt_data": prompt_data,
                "needs_processing": needs_processing
            })
            
            if needs_processing:
                # Format the prompt
                unformatted_prompt = concatenate_prompt(
                    prompt_data["task"],
                    prompt_data["rules"],
                    prompt_data["grader"]
                )
                messages = format_chat_messages(unformatted_prompt)
                prompt_string = messages_to_prompt_string(messages, tokenizer=self.tokenizer)
                
                # Create N_INITIAL_ROLLOUT_TESTS copies for each rollout
                for rollout_idx in range(num_initial_rollouts):
                    for test_idx in range(self.n_initial_rollout_tests):
                        all_prompts.append(prompt_string)
                        metadata_map.append((prompt_idx, rollout_idx, test_idx))
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'prompts_prepared': len([p for p in prompt_info_list if p['needs_processing']]),
                    'total_requests': len(all_prompts)
                })
        
        if pbar:
            pbar.close()
        
        if not HAS_TQDM:
            print(f"  ✓ Prepared {len(all_prompts)} initial rollout requests for {len([p for p in prompt_info_list if p['needs_processing']])} prompts")
        
        return all_prompts, metadata_map, prompt_info_list
    
    def process_initial_rollouts(
        self,
        all_outputs: List[Any],
        metadata_map: List[tuple[int, int, int]],
        prompt_info_list: List[Dict[str, Any]],
        num_initial_rollouts: int
    ) -> List[Dict[str, Any]]:
        """
        Process initial rollout outputs, select best rollouts, and prepare for counterfactual generation.
        
        Args:
            all_outputs: List of vLLM output objects
            metadata_map: List of (prompt_idx, rollout_idx, test_idx) tuples
            prompt_info_list: List of prompt info dicts
            num_initial_rollouts: Number of initial rollouts per prompt
        
        Returns:
            List of processed prompt data dicts with keys: prompt_idx, prompt_name, prompt_data,
            rollouts_data (list of dicts with rollout_idx, best_rollout, sentences, selected_indices)
        """
        # Group outputs by (prompt_idx, rollout_idx, test_idx)
        rollouts_by_prompt = defaultdict(lambda: defaultdict(list))
        
        for i, output in enumerate(all_outputs):
            if i < len(metadata_map):
                prompt_idx, rollout_idx, test_idx = metadata_map[i]
                if output.outputs and len(output.outputs) > 0:
                    text = output.outputs[0].text
                    if text:
                        rollouts_by_prompt[prompt_idx][rollout_idx].append(text)
        
        processed_prompts = []
        
        if HAS_TQDM:
            pbar = tqdm(total=len([p for p in prompt_info_list if p['needs_processing']]), 
                       desc="Phase 2: Processing initial rollouts", unit="prompt")
        else:
            pbar = None
            print("Phase 2: Processing initial rollouts...")
        
        for prompt_info in prompt_info_list:
            if not prompt_info['needs_processing']:
                continue
            
            prompt_idx = prompt_info['prompt_idx']
            prompt_data = prompt_info['prompt_data']
            prompt_name = prompt_info['prompt_name']
            
            rollouts_data = []
            
            # Process each rollout for this prompt
            for rollout_idx in range(num_initial_rollouts):
                test_rollouts = rollouts_by_prompt[prompt_idx].get(rollout_idx, [])
                
                if not test_rollouts:
                    continue
                
                # Calculate sentence counts for each test rollout
                rollout_sentence_counts = []
                for rollout in test_rollouts:
                    sentences = split_sentences(rollout)
                    rollout_sentence_counts.append((len(sentences), rollout))
                
                if not rollout_sentence_counts:
                    continue
                
                sentence_counts = [count for count, _ in rollout_sentence_counts]
                
                # Select best rollout based on strategy
                if self.initial_rollout_selection_strategy == "MIN":
                    best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
                elif self.initial_rollout_selection_strategy == "MEDIAN":
                    sentence_counts_sorted = sorted(sentence_counts)
                    median_count = sentence_counts_sorted[len(sentence_counts_sorted) // 2]
                    best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: abs(x[0] - median_count))
                else:
                    best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
                
                # Split into sentences
                sentences = split_sentences(best_rollout)
                total_sentences = len(sentences)
                
                # Determine which sentence indices to process
                if self.num_sentences_to_sample is None or self.num_sentences_to_sample >= total_sentences:
                    selected_indices = list(range(total_sentences))
                else:
                    num_to_sample = self.num_sentences_to_sample
                    if total_sentences == 1:
                        selected_indices = [0]
                    else:
                        step = (total_sentences - 1) / (num_to_sample - 1)
                        selected_indices = [int(round(i * step)) for i in range(num_to_sample)]
                        selected_indices = sorted(list(set(selected_indices)))
                        if selected_indices[0] != 0:
                            selected_indices.insert(0, 0)
                        if selected_indices[-1] != total_sentences - 1:
                            selected_indices.append(total_sentences - 1)
                
                rollouts_data.append({
                    "rollout_idx": rollout_idx,
                    "best_rollout": best_rollout,
                    "sentences": sentences,
                    "selected_indices": selected_indices,
                    "total_sentences": total_sentences
                })
            
            if rollouts_data:
                processed_prompts.append({
                    "prompt_idx": prompt_idx,
                    "prompt_name": prompt_name,
                    "prompt_data": prompt_data,
                    "rollouts_data": rollouts_data
                })
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({'processed': len(processed_prompts)})
        
        if pbar:
            pbar.close()
        
        if not HAS_TQDM:
            print(f"  ✓ Processed {len(processed_prompts)} prompts")
        
        return processed_prompts
    
    def save_initial_rollouts_checkpoint(
        self,
        processed_prompts: List[Dict[str, Any]],
        output_dir: str
    ) -> None:
        """
        Save initial rollouts as a checkpoint file for recovery.
        This allows resuming even if counterfactual generation fails.
        
        Args:
            processed_prompts: List of processed prompt data dicts
            output_dir: Directory to save checkpoint
        """
        checkpoint_path = Path(output_dir) / "_checkpoint_initial_rollouts.json"
        try:
            checkpoint_data = []
            for prompt in processed_prompts:
                checkpoint_data.append({
                    "prompt_idx": prompt["prompt_idx"],
                    "prompt_name": prompt["prompt_name"],
                    "rollouts_data": [
                        {
                            "rollout_idx": rd["rollout_idx"],
                            "best_rollout": rd["best_rollout"],
                            "total_sentences": rd["total_sentences"],
                            "selected_indices": rd["selected_indices"]
                        }
                        for rd in prompt["rollouts_data"]
                    ]
                })
            
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.replace(checkpoint_path)
            
            if not HAS_TQDM:
                print(f"  ✓ Saved initial rollouts checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not save checkpoint: {e}")
    
    def load_initial_rollouts_checkpoint(
        self,
        output_dir: str,
        prompts_to_process: List[Dict[str, str]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Load initial rollouts from checkpoint if available.
        
        Args:
            output_dir: Directory to load checkpoint from
            prompts_to_process: Original prompts list to match against
        
        Returns:
            List of processed prompt data dicts if checkpoint exists, None otherwise
        """
        checkpoint_path = Path(output_dir) / "_checkpoint_initial_rollouts.json"
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            # Reconstruct processed_prompts structure
            processed_prompts = []
            prompt_idx_to_data = {p.get("name", f"prompt_{i}"): p for i, p in enumerate(prompts_to_process)}
            
            for checkpoint_item in checkpoint_data:
                prompt_name = checkpoint_item["prompt_name"]
                prompt_data = prompt_idx_to_data.get(prompt_name)
                
                if prompt_data is None:
                    continue
                
                # Reconstruct rollouts_data with sentences
                rollouts_data = []
                for rd_checkpoint in checkpoint_item["rollouts_data"]:
                    best_rollout = rd_checkpoint["best_rollout"]
                    sentences = split_sentences(best_rollout)
                    
                    rollouts_data.append({
                        "rollout_idx": rd_checkpoint["rollout_idx"],
                        "best_rollout": best_rollout,
                        "sentences": sentences,
                        "selected_indices": rd_checkpoint["selected_indices"],
                        "total_sentences": rd_checkpoint["total_sentences"]
                    })
                
                processed_prompts.append({
                    "prompt_idx": checkpoint_item["prompt_idx"],
                    "prompt_name": prompt_name,
                    "prompt_data": prompt_data,
                    "rollouts_data": rollouts_data
                })
            
            if processed_prompts:
                if not HAS_TQDM:
                    print(f"  ✓ Loaded initial rollouts checkpoint ({len(processed_prompts)} prompts)")
                return processed_prompts
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not load checkpoint: {e}")
        
        return None
    
    def prepare_all_counterfactuals(
        self,
        processed_prompts: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        skip_completed: bool = True
    ) -> tuple[List[str], List[tuple[int, int, int]]]:
        """
        Prepare all counterfactual prompts from processed prompt data.
        Optionally skips sentences that are already completed.
        
        Args:
            processed_prompts: List of processed prompt data dicts
            output_dir: Directory to check for existing results (if skip_completed=True)
            skip_completed: If True, skip sentences that already have complete results
        
        Returns:
            Tuple of (all_prompts_list, metadata_map)
            - all_prompts_list: List of counterfactual prompt strings
            - metadata_map: List of (prompt_idx, rollout_idx, sentence_idx) tuples
        """
        all_prompts = []
        metadata_map = []
        prompt_idx_to_name = {p["prompt_idx"]: p["prompt_name"] for p in processed_prompts}
        
        # Get completed sentences if skipping
        completed_sentences = set()
        if skip_completed and output_dir:
            for processed_prompt in processed_prompts:
                prompt_idx = processed_prompt["prompt_idx"]
                prompt_name = prompt_idx_to_name[prompt_idx]
                
                for rollout_data in processed_prompt["rollouts_data"]:
                    rollout_idx = rollout_data["rollout_idx"]
                    output_path = self._get_output_path(prompt_name, rollout_idx, output_dir)
                    existing = self._load_existing_result(output_path)
                    
                    if existing is not None:
                        # Get set of completed sentence indices
                        existing_sentence_data = existing.get("sentence_data", [])
                        existing_indices = {s.get("sentence_index") for s in existing_sentence_data if isinstance(s, dict)}
                        for sent_idx in existing_indices:
                            completed_sentences.add((prompt_idx, rollout_idx, sent_idx))
        
        if HAS_TQDM:
            total_prompts = len(processed_prompts)
            pbar = tqdm(total=total_prompts, desc="Phase 3: Preparing counterfactual prompts", unit="prompt")
        else:
            pbar = None
            print("Phase 3: Preparing counterfactual prompts...")
        
        skipped_count = 0
        
        for processed_prompt in processed_prompts:
            prompt_data = processed_prompt["prompt_data"]
            prompt_idx = processed_prompt["prompt_idx"]
            
            # Pre-format base prompt once
            unformatted_prompt = concatenate_prompt(
                prompt_data["task"],
                prompt_data["rules"],
                prompt_data["grader"]
            )
            base_messages = format_chat_messages(unformatted_prompt)
            base_prompt_string = messages_to_prompt_string(base_messages, tokenizer=self.tokenizer)
            
            # For each rollout
            for rollout_data in processed_prompt["rollouts_data"]:
                rollout_idx = rollout_data["rollout_idx"]
                sentences = rollout_data["sentences"]
                selected_indices = rollout_data["selected_indices"]
                
                # For each selected sentence
                for sent_idx in selected_indices:
                    # Skip if already completed
                    if (prompt_idx, rollout_idx, sent_idx) in completed_sentences:
                        skipped_count += 1  # Count unique prompts skipped (not completions)
                        continue
                    
                    prefix = get_sentence_prefix(sentences, sent_idx)
                    
                    # Build counterfactual prompt: base + assistant prefix
                    counterfactual_messages = base_messages.copy()
                    counterfactual_messages.append({"role": "assistant", "content": prefix})
                    counterfactual_prompt = messages_to_prompt_string(counterfactual_messages, tokenizer=self.tokenizer)
                    
                    # Add num_counterfactuals copies (duplicate prompts for better batching)
                    for _ in range(self.num_counterfactuals):
                        all_prompts.append(counterfactual_prompt)
                        metadata_map.append((prompt_idx, rollout_idx, sent_idx))
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'total_requests': len(all_prompts),
                    'skipped': skipped_count
                })
        
        if pbar:
            pbar.close()
        
        if not HAS_TQDM:
            print(f"  ✓ Prepared {len(all_prompts)} counterfactual requests (skipped {skipped_count} already completed)")
        
        return all_prompts, metadata_map
    
    def _merge_existing_results(
        self,
        results_dict: Dict[int, Dict[int, List[Dict[str, Any]]]],
        processed_prompts: List[Dict[str, Any]],
        output_dir: str
    ) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
        """
        Merge existing results from files into results_dict.
        This allows resuming from partial results.
        
        Args:
            results_dict: Current results dict (may be empty or partial)
            processed_prompts: List of processed prompt data dicts
            output_dir: Directory to load existing results from
        
        Returns:
            Merged results dict
        """
        prompt_idx_to_name = {p["prompt_idx"]: p["prompt_name"] for p in processed_prompts}
        
        for processed_prompt in processed_prompts:
            prompt_idx = processed_prompt["prompt_idx"]
            prompt_name = prompt_idx_to_name[prompt_idx]
            
            for rollout_data in processed_prompt["rollouts_data"]:
                rollout_idx = rollout_data["rollout_idx"]
                output_path = self._get_output_path(prompt_name, rollout_idx, output_dir)
                
                existing = self._load_existing_result(output_path)
                if existing is not None:
                    # Merge existing results
                    existing_sentence_data = existing.get("sentence_data", [])
                    existing_dict = {s.get("sentence_index"): s for s in existing_sentence_data if isinstance(s, dict)}
                    
                    # If we already have results for this prompt/rollout, merge them
                    if prompt_idx in results_dict and rollout_idx in results_dict[prompt_idx]:
                        current_dict = {s.get("sentence_index"): s for s in results_dict[prompt_idx][rollout_idx] if isinstance(s, dict)}
                        # Update with existing (existing takes precedence for completed sentences)
                        current_dict.update(existing_dict)
                        results_dict[prompt_idx][rollout_idx] = [current_dict[i] for i in sorted(current_dict.keys())]
                    else:
                        # Use existing results
                        results_dict[prompt_idx][rollout_idx] = existing_sentence_data
        
        return results_dict
    
    def process_all_counterfactuals(
        self,
        all_outputs: List[Any],
        metadata_map: List[tuple[int, int, int]],
        processed_prompts: List[Dict[str, Any]],
        output_dir: str,
        save_progressively: bool = True
    ) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
        """
        Process counterfactual outputs and create sentence data structures.
        Optionally saves results incrementally as they're processed.
        
        Args:
            all_outputs: List of vLLM output objects
            metadata_map: List of (prompt_idx, rollout_idx, sentence_idx) tuples
            processed_prompts: List of processed prompt data dicts
            output_dir: Directory to save output files
            save_progressively: If True, save results as they're processed (default: True)
        
        Returns:
            Dict[prompt_idx][rollout_idx] = List[sentence_data]
        """
        # Group outputs by (prompt_idx, rollout_idx, sentence_idx)
        # Each output contains one completion (prompts are duplicated instead of using n=)
        rollouts_by_key = defaultdict(list)
        
        for i, output in enumerate(all_outputs):
            if i < len(metadata_map):
                prompt_idx, rollout_idx, sentence_idx = metadata_map[i]
                # output.outputs is a list with one completion (since we duplicate prompts)
                if output.outputs and len(output.outputs) > 0:
                    # Get the single completion from this output
                    completion = output.outputs[0]
                    text = completion.text
                    if text:
                        rollouts_by_key[(prompt_idx, rollout_idx, sentence_idx)].append(text)
        
        # Start by loading any existing partial results
        results_dict = defaultdict(lambda: defaultdict(list))
        results_dict = self._merge_existing_results(results_dict, processed_prompts, output_dir)
        
        prompt_idx_to_name = {p["prompt_idx"]: p["prompt_name"] for p in processed_prompts}
        
        if HAS_TQDM:
            total_prompts = len(processed_prompts)
            pbar = tqdm(total=total_prompts, desc="Phase 5: Processing counterfactuals", unit="prompt")
        else:
            pbar = None
            print("Phase 5: Processing counterfactuals...")
        
        for processed_prompt in processed_prompts:
            prompt_idx = processed_prompt["prompt_idx"]
            prompt_name = prompt_idx_to_name[prompt_idx]
            prompt_data = processed_prompt["prompt_data"]
            
            # Pre-format prompt messages once
            unformatted_prompt = concatenate_prompt(
                prompt_data["task"],
                prompt_data["rules"],
                prompt_data["grader"]
            )
            formatted_prompt = format_chat_messages(unformatted_prompt)
            
            # Process each rollout
            for rollout_data in processed_prompt["rollouts_data"]:
                rollout_idx = rollout_data["rollout_idx"]
                sentences = rollout_data["sentences"]
                selected_indices = rollout_data["selected_indices"]
                
                # Process each selected sentence
                for sent_idx in selected_indices:
                    rollouts = rollouts_by_key.get((prompt_idx, rollout_idx, sent_idx), [])
                    
                    # Detect reward hacking in each rollout
                    reward_hacks = [detect_reward_hacking(r) for r in rollouts]
                    p_reward_hacks = sum(reward_hacks) / len(rollouts) if rollouts else 0.0
                    
                    sentence_data = create_sentence_data(
                        sentence_index=sent_idx,
                        sentence=sentences[sent_idx],
                        p_reward_hacks=p_reward_hacks,
                        rollouts=rollouts,
                        formatted_prompt=formatted_prompt,
                        unformatted_prompt=unformatted_prompt
                    )
                    results_dict[prompt_idx][rollout_idx].append(sentence_data)
                
                # Save this rollout's results incrementally
                if save_progressively:
                    sentence_data_list = results_dict[prompt_idx][rollout_idx]
                    output_path = self._get_output_path(prompt_name, rollout_idx, output_dir)
                    # Get full_rollout_text from rollout_data
                    full_rollout_text = rollout_data.get("best_rollout")
                    success = self._save_result_file(output_path, sentence_data_list, full_rollout_text=full_rollout_text)
                    if not success and pbar:
                        pbar.write(f"  ⚠ Failed to save {prompt_name}_rollout_{rollout_idx}")
            
            if pbar:
                pbar.update(1)
                total_sentences = sum(len(results_dict[prompt_idx][r]) for r in results_dict[prompt_idx])
                pbar.set_postfix({'sentences_processed': total_sentences})
        
        if pbar:
            pbar.close()
        
        if not HAS_TQDM:
            total_sentences = sum(len(sentences) for prompt_dict in results_dict.values() for sentences in prompt_dict.values())
            print(f"  ✓ Processed {total_sentences} sentences across {len(results_dict)} prompts")
        
        return results_dict
    
    def save_all_results(
        self,
        results_dict: Dict[int, Dict[int, List[Dict[str, Any]]]],
        processed_prompts: List[Dict[str, Any]],
        output_dir: str,
        force_resave: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Save all results to JSON files.
        If results were already saved progressively, this just verifies/updates them.
        
        Args:
            results_dict: Dict[prompt_idx][rollout_idx] = List[sentence_data]
            processed_prompts: List of processed prompt data dicts
            output_dir: Directory to save output files
            force_resave: If True, resave even if file exists (default: False)
        
        Returns:
            List of result summary dicts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a mapping from prompt_idx to prompt_name
        prompt_idx_to_name = {p["prompt_idx"]: p["prompt_name"] for p in processed_prompts}
        
        # Count total files to save
        total_files = sum(len(rollouts) for prompt_dict in results_dict.values() for rollouts in prompt_dict.values())
        
        if HAS_TQDM:
            pbar = tqdm(total=total_files, desc="Phase 6: Verifying/saving results", unit="file")
        else:
            pbar = None
            print("Phase 6: Verifying/saving results...")
        
        saved_results = []
        skipped = 0
        
        for prompt_idx, rollouts_dict in results_dict.items():
            prompt_name = prompt_idx_to_name.get(prompt_idx, f"prompt_{prompt_idx}")
            
            for rollout_idx, sentence_data_list in rollouts_dict.items():
                output_path = self._get_output_path(prompt_name, rollout_idx, output_dir)
                
                # Skip if already exists and not forcing resave
                if not force_resave and output_path.exists():
                    existing = self._load_existing_result(output_path)
                    if existing is not None:
                        existing_sentence_data = existing.get("sentence_data", [])
                        if len(existing_sentence_data) == len(sentence_data_list):
                            skipped += 1
                            saved_results.append({
                                "prompt_name": prompt_name,
                                "rollout_index": rollout_idx,
                                "num_sentences": len(sentence_data_list),
                                "sentence_data": existing_sentence_data  # Use existing data
                            })
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix({'saved': len(saved_results), 'skipped': skipped})
                        continue
                
                # Get full_rollout_text from processed_prompts
                full_rollout_text = None
                for processed_prompt in processed_prompts:
                    if processed_prompt["prompt_idx"] == prompt_idx:
                        for rollout_data in processed_prompt["rollouts_data"]:
                            if rollout_data["rollout_idx"] == rollout_idx:
                                full_rollout_text = rollout_data.get("best_rollout")
                                break
                        if full_rollout_text is not None:
                            break
                
                # Save the file
                success = self._save_result_file(output_path, sentence_data_list, full_rollout_text=full_rollout_text)
                if success:
                    saved_results.append({
                        "prompt_name": prompt_name,
                        "rollout_index": rollout_idx,
                        "num_sentences": len(sentence_data_list),
                        "sentence_data": sentence_data_list
                    })
                else:
                    if pbar:
                        pbar.write(f"  ⚠ Failed to save {prompt_name}_rollout_{rollout_idx}")
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({'saved': len(saved_results), 'skipped': skipped})
        
        if pbar:
            pbar.close()
        
        if not HAS_TQDM:
            print(f"  ✓ Saved/verified {len(saved_results)} result files ({skipped} already existed)")
        
        return saved_results
    
    
    def process_prompt_offline(
        self,
        prompt_data: Dict[str, str],
        output_dir: str = "src/rollouts",
        skip_if_exists: bool = True,
        prompt_pbar: Optional[Any] = None,
        rollout_idx: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single prompt using offline batched inference.
        
        This is the new batched approach that generates all counterfactuals
        for all sentences in one megabatch call to vLLM.
        
        Args:
            prompt_data: Prompt data dict
            output_dir: Directory to save output
            skip_if_exists: Whether to skip if output file exists
            prompt_pbar: Optional tqdm progress bar to update for sentence progress
        
        Returns:
            List of sentence data dictionaries
        """
        if self.llm is None or self.sampling_params is None:
            raise ValueError("LLM engine and sampling_params must be set before processing prompts")
        
        prompt_name = prompt_data.get("name", "unknown")
        if rollout_idx is not None:
            output_path = Path(output_dir) / f"{prompt_name}_rollout_{rollout_idx}_rollouts.json"
        else:
            output_path = Path(output_dir) / f"{prompt_name}_rollouts.json"
        
        # Check if output already exists
        if skip_if_exists and output_path.exists():
            if prompt_pbar:
                prompt_pbar.set_postfix({'status': 'checking existing file...'})
            if not HAS_TQDM:
                print(f"\n  Skipping {prompt_name} (output already exists)")
            try:
                with open(output_path) as f:
                    data = json.load(f)
                    if prompt_pbar:
                        prompt_pbar.total = len(data)
                        prompt_pbar.n = len(data)
                        prompt_pbar.refresh()
                        prompt_pbar.set_postfix({'status': 'loaded from file'})
                    return data
            except:
                pass  # If file is corrupted, regenerate
        
        # Step 1: Generate initial rollouts
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': f'generating {self.n_initial_rollout_tests} initial rollouts...'})
        elif not HAS_TQDM:
            print(f"\nProcessing prompt: {prompt_name}")
            print(f"  Step 1: Generating {self.n_initial_rollout_tests} initial rollouts...")
        
        unformatted_prompt = concatenate_prompt(
            prompt_data["task"],
            prompt_data["rules"],
            prompt_data["grader"]
        )
        messages = format_chat_messages(unformatted_prompt)
        # Get tokenizer from LLM for proper chat formatting (cache it)
        tokenizer = None
        try:
            tokenizer = getattr(self.llm, 'llm_engine', None)
            if tokenizer:
                tokenizer = getattr(tokenizer, 'tokenizer', None)
        except:
            pass
        prompt_string = messages_to_prompt_string(messages, tokenizer=tokenizer)
        
        # Generate N initial rollouts in one batch
        initial_prompts = [prompt_string] * self.n_initial_rollout_tests
        initial_outputs = self.llm.generate(initial_prompts, self.sampling_params)
        
        initial_rollouts = []
        for output in initial_outputs:
            if output.outputs and len(output.outputs) > 0:
                text = output.outputs[0].text
                if text:
                    initial_rollouts.append(text)
        
        if not initial_rollouts:
            if prompt_pbar:
                prompt_pbar.set_postfix({'status': 'failed to generate initial rollouts'})
            if not HAS_TQDM:
                print(f"  ✗ Failed to generate initial rollouts for {prompt_name}")
            return []
        
        # Step 2: Select best rollout based on strategy
        if prompt_pbar:
            strategy_name = self.initial_rollout_selection_strategy.lower()
            prompt_pbar.set_postfix({'status': f'calculating {strategy_name} sentence length...'})
        elif not HAS_TQDM:
            print(f"  ✓ Generated {len(initial_rollouts)} initial rollouts")
            print(f"  Step 2: Calculating {self.initial_rollout_selection_strategy} sentence length...")
        
        # Calculate sentence counts for each rollout
        rollout_sentence_counts = []
        for rollout in initial_rollouts:
            sentences = split_sentences(rollout)
            rollout_sentence_counts.append((len(sentences), rollout))
        
        sentence_counts = [count for count, _ in rollout_sentence_counts]
        
        # Select rollout based on strategy
        if self.initial_rollout_selection_strategy == "MIN":
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
            strategy_display = "minimum"
        elif self.initial_rollout_selection_strategy == "MEDIAN":
            sentence_counts_sorted = sorted(sentence_counts)
            median_count = sentence_counts_sorted[len(sentence_counts_sorted) // 2]
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: abs(x[0] - median_count))
            strategy_display = f"median (target: {median_count})"
        else:
            best_count, best_rollout = min(rollout_sentence_counts, key=lambda x: x[0])
            strategy_display = "minimum (invalid strategy, defaulted)"
        
        if prompt_pbar:
            prompt_pbar.set_postfix({
                'status': f'selected rollout with {best_count} sentences ({strategy_display})',
                'sentence_counts': f'{min(sentence_counts)}-{max(sentence_counts)}'
            })
        elif not HAS_TQDM:
            print(f"  ✓ Sentence counts: {min(sentence_counts)}-{max(sentence_counts)}")
            print(f"  ✓ Selected rollout with {best_count} sentences ({strategy_display})")
        
        # Step 3: Split into sentences
        sentences = split_sentences(best_rollout)
        total_sentences = len(sentences)
        
        # Determine which sentence indices to process
        if self.num_sentences_to_sample is None or self.num_sentences_to_sample >= total_sentences:
            # Process all sentences (backward compatible)
            selected_indices = list(range(total_sentences))
            sampling_info = f"all {total_sentences}"
        else:
            # Evenly sample N sentences
            num_to_sample = self.num_sentences_to_sample
            if total_sentences == 1:
                selected_indices = [0]
            else:
                # Calculate evenly-spaced indices
                step = (total_sentences - 1) / (num_to_sample - 1)
                selected_indices = [int(round(i * step)) for i in range(num_to_sample)]
                # Ensure we don't have duplicates and cover the range
                selected_indices = sorted(list(set(selected_indices)))
                # Ensure first and last are included
                if selected_indices[0] != 0:
                    selected_indices.insert(0, 0)
                if selected_indices[-1] != total_sentences - 1:
                    selected_indices.append(total_sentences - 1)
            sampling_info = f"{len(selected_indices)} of {total_sentences}"
        
        if prompt_pbar:
            prompt_pbar.total = len(selected_indices)
            prompt_pbar.n = 0
            prompt_pbar.refresh()
            prompt_pbar.set_postfix({'status': f'preparing megabatch for {sampling_info} sentences...'})
        elif not HAS_TQDM:
            print(f"  ✓ Found {total_sentences} sentence(s), processing {sampling_info}")
            print("  Step 3: Preparing megabatch for counterfactual generation...")
        
        # Step 4: Prepare the megabatch - counterfactuals for selected sentences
        all_prompts = []
        metadata_map = []  # Maps prompt index to sentence_idx
        
        # Get tokenizer once (cached for efficiency)
        tokenizer = None
        try:
            tokenizer = getattr(self.llm, 'llm_engine', None)
            if tokenizer:
                tokenizer = getattr(tokenizer, 'tokenizer', None)
        except:
            pass
        
        for sent_idx in selected_indices:
            prefix = get_sentence_prefix(sentences, sent_idx)
            
            # Create messages with prefix for this sentence
            counterfactual_messages = format_chat_messages(unformatted_prompt)
            counterfactual_messages.append({"role": "assistant", "content": prefix})
            counterfactual_prompt = messages_to_prompt_string(counterfactual_messages, tokenizer=tokenizer)
            
            # Add num_counterfactuals copies of this prompt
            for _ in range(self.num_counterfactuals):
                all_prompts.append(counterfactual_prompt)
                metadata_map.append(sent_idx)
        
        # Step 5: The megabatch generation - ONE call to vLLM
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': f'generating {len(all_prompts)} rollouts in megabatch ({sampling_info} sentences)...'})
        elif not HAS_TQDM:
            print(f"  Step 4: Generating {len(all_prompts)} rollouts in megabatch ({sampling_info} sentences)...")
        
        outputs = self.llm.generate(all_prompts, self.sampling_params)
        
        # Step 6: Reassemble data - map outputs back to sentences
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': 'reassembling data...'})
        elif not HAS_TQDM:
            print("  Step 5: Reassembling data...")
        
        # Group rollouts by sentence index
        sentence_rollouts = defaultdict(list)
        for i, output in enumerate(outputs):
            sent_idx = metadata_map[i]
            if output.outputs and len(output.outputs) > 0:
                generated_text = output.outputs[0].text
                if generated_text:
                    sentence_rollouts[sent_idx].append(generated_text)
        
        # Step 7: Create sentence data structures
        sentence_data_list = []
        formatted_prompt = format_chat_messages(unformatted_prompt)
        
        for sent_idx in selected_indices:
            rollouts = sentence_rollouts.get(sent_idx, [])
            
            # Detect reward hacking in each rollout
            reward_hacks = [detect_reward_hacking(r) for r in rollouts]
            p_reward_hacks = sum(reward_hacks) / len(rollouts) if rollouts else 0.0
            
            sentence_data = create_sentence_data(
                sentence_index=sent_idx,  # Keep original index for traceability
                sentence=sentences[sent_idx],
                p_reward_hacks=p_reward_hacks,
                rollouts=rollouts,
                formatted_prompt=formatted_prompt,
                unformatted_prompt=unformatted_prompt
            )
            sentence_data_list.append(sentence_data)
            
            # Update progress bar
            if prompt_pbar:
                prompt_pbar.update(1)
                completed = len(sentence_data_list)
                if completed > 0:
                    avg_p_hacks = sum(s.get("p_reward_hacks", 0) for s in sentence_data_list) / completed
                    prompt_pbar.set_postfix({
                        'sentences': f'{completed}/{len(selected_indices)}',
                        'avg_p_hacks': f'{avg_p_hacks:.3f}'
                    })
        
        # Step 8: Save to file
        if prompt_pbar:
            prompt_pbar.set_postfix({'status': 'saving...'})
        elif not HAS_TQDM:
            print(f"\n  Step 6: Saving to {output_path}...")
        
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sentence_data_list, f, indent=2)
        
        if prompt_pbar:
            avg_p_hacks = sum(s.get("p_reward_hacks", 0) for s in sentence_data_list) / len(sentence_data_list) if sentence_data_list else 0
            max_p_hacks = max((s.get("p_reward_hacks", 0) for s in sentence_data_list), default=0)
            prompt_pbar.set_postfix({
                'status': 'complete',
                'avg_p_hacks': f'{avg_p_hacks:.3f}',
                'max_p_hacks': f'{max_p_hacks:.3f}'
            })
        elif not HAS_TQDM:
            print(f"  ✓ Saved {len(sentence_data_list)} sentence data entries")
        
        return sentence_data_list
    
    def run(self, prompts_path: str, output_dir: str = "src/rollouts"):
        """
        Run the full pipeline on all prompts.
        
        Args:
            prompts_path: Path to prompts JSON file
            output_dir: Directory to save output files
        """
        # Load prompts
        print("Loading prompts...")
        prompts = self.load_prompts(prompts_path)
        print(f"Loaded {len(prompts)} prompt(s)")
        
        # Process each prompt
        for prompt_data in prompts:
            self.process_prompt_offline(prompt_data, output_dir)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate counterfactual rollouts for reward hacking analysis")
    parser.add_argument(
        "--prompts",
        type=str,
        default="src/data/reward_hacking_prompts.json",
        help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/rollouts",
        help="Output directory for rollout files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model name"
    )
    parser.add_argument(
        "--num-counterfactuals",
        type=int,
        default=50,
        help="Number of counterfactual rollouts per sentence"
    )
    
    args = parser.parse_args()
    
    # Note: This main() function is a legacy entry point. For full functionality,
    # use run_experiment.py which properly initializes vLLM.
    # This will fail at runtime because llm, sampling_params, and tokenizer are not set.
    generator = RolloutGenerator(model=args.model)
    generator.num_counterfactuals = args.num_counterfactuals
    generator.run(args.prompts, args.output_dir)


if __name__ == "__main__":
    main()

