"""
Utilities for reconstructing sequences and formatting for activation extraction.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizer

# Add src directory to path to import from utils
# Path from src/activations/utils.py -> src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_utils import get_sentence_prefix


def reconstruct_sequence_messages(sentence_data_list: List[Dict[str, Any]], sentence_index: int, 
                                  full_rollout_text: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Reconstruct chat messages up to a given sentence index.
    
    This function handles both sequential and selected sentence formats:
    - Sequential: sentence_data_list contains all sentences in order (0, 1, 2, ...)
    - Selected: sentence_data_list only contains selected sentences (e.g., [0, 5, 11, 16, ...])
                 Each entry has a 'sentence_index' field indicating its position in the original rollout.
    
    Args:
        sentence_data_list: List of sentence data dictionaries from rollout JSON
        sentence_index: Index of the sentence to include (0-based, refers to original sentence position)
        full_rollout_text: Optional full rollout text for accurate reconstruction (new format)
    
    Returns:
        List of chat messages: [user_prompt, assistant_response_up_to_index]
    """
    if not sentence_data_list:
        raise ValueError("sentence_data_list cannot be empty")
    
    # Get the formatted prompt from the first sentence (should be same for all)
    first_sentence = sentence_data_list[0]
    formatted_prompt = first_sentence.get("formatted_prompt", [])
    
    if not formatted_prompt:
        raise ValueError("formatted_prompt not found in sentence data")
    
    # If full_rollout_text is available, use it for accurate reconstruction
    if full_rollout_text is not None:
        # Import here to avoid circular imports
        from utils.text_utils import split_sentences
        
        # Split full rollout into all sentences
        all_sentences = split_sentences(full_rollout_text)
        
        # Verify sentence_index is valid
        if sentence_index < 0 or sentence_index >= len(all_sentences):
            raise ValueError(
                f"sentence_index {sentence_index} out of range [0, {len(all_sentences)}) "
                f"for full rollout with {len(all_sentences)} sentences"
            )
        
        # Get prefix up to and including target sentence
        sentences = all_sentences[:sentence_index + 1]
        target_pos_in_filtered = sentence_index
    
    else:
        # Fall back to old behavior (may have gaps in selected sentences format)
        import warnings
        warnings.warn(
            "full_rollout_text not available. Reconstructing from stored sentences only. "
            "This may have gaps if using selected sentences format.",
            UserWarning
        )
        
        # Check if we're using selected sentences format (each entry has sentence_index field)
        # vs sequential format (indices match list positions)
        has_sentence_index_field = all("sentence_index" in s for s in sentence_data_list)
        
        if has_sentence_index_field:
            # Selected sentences format: filter by sentence_index field
            # Get all sentences where sentence_index <= target sentence_index
            relevant_sentences = [
                s for s in sentence_data_list 
                if s.get("sentence_index", -1) <= sentence_index
            ]
            
            # Sort by sentence_index to ensure correct order
            relevant_sentences.sort(key=lambda s: s.get("sentence_index", -1))
            
            # Verify the target sentence exists
            target_sentence = next(
                (s for s in relevant_sentences if s.get("sentence_index") == sentence_index),
                None
            )
            if target_sentence is None:
                raise ValueError(
                    f"sentence_index {sentence_index} not found in sentence_data_list. "
                    f"Available indices: {[s.get('sentence_index') for s in sentence_data_list]}"
                )
            
            # Extract sentences in order
            sentences = [s["sentence"] for s in relevant_sentences]
            
            # Find the position of target sentence in the filtered list
            target_pos_in_filtered = len(relevant_sentences) - 1  # It's the last one by construction
            
        else:
            # Sequential format: use list indices directly
            if sentence_index < 0 or sentence_index >= len(sentence_data_list):
                raise ValueError(f"sentence_index {sentence_index} out of range [0, {len(sentence_data_list)})")
            
            sentences = [s["sentence"] for s in sentence_data_list[:sentence_index + 1]]
            target_pos_in_filtered = sentence_index
    
    # Concatenate sentences to form assistant response
    assistant_content = get_sentence_prefix(sentences, target_pos_in_filtered)
    
    # Construct messages: [user_prompt, assistant_response]
    messages = formatted_prompt.copy()  # Start with user prompt
    messages.append({"role": "assistant", "content": assistant_content})
    
    return messages


def format_with_chat_template(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizer) -> str:
    """
    Format chat messages with tokenizer's chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Tokenizer with apply_chat_template method
    
    Returns:
        Formatted string with chat template applied
    """
    if not hasattr(tokenizer, 'apply_chat_template'):
        raise ValueError("Tokenizer does not have apply_chat_template method")
    
    chat_formatted = tokenizer.apply_chat_template(
        messages,  # type: ignore[arg-type]
        tokenize=False,
        add_generation_prompt=True  # Match generation format
    )
    
    if not isinstance(chat_formatted, str):
        raise TypeError(f"Expected string from apply_chat_template, got {type(chat_formatted)}")
    
    return chat_formatted