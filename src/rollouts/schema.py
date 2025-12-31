"""
Data schema definitions for rollout output.
"""

from typing import List, Dict, Any, TypedDict


class SentenceData(TypedDict, total=False):
    """Data structure for a single sentence with counterfactual rollouts."""
    sentence_index: int
    sentence: str
    p_reward_hacks: float
    rollouts: List[str]  # The 50 generated continuations
    formatted_prompt: List[Dict[str, str]]  # Chat messages format
    unformatted_prompt: str  # Raw concatenated prompt


def validate_sentence_data(data: Dict[str, Any]) -> bool:
    """
    Validate that a sentence data dict has all required fields.
    
    Args:
        data: Dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["sentence_index", "sentence", "p_reward_hacks", "rollouts", 
                      "formatted_prompt", "unformatted_prompt"]
    
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate types
    if not isinstance(data["sentence_index"], int):
        return False
    if not isinstance(data["sentence"], str):
        return False
    if not isinstance(data["p_reward_hacks"], (int, float)):
        return False
    if not isinstance(data["rollouts"], list):
        return False
    if not isinstance(data["formatted_prompt"], list):
        return False
    if not isinstance(data["unformatted_prompt"], str):
        return False
    
    return True


def create_sentence_data(
    sentence_index: int,
    sentence: str,
    p_reward_hacks: float,
    rollouts: List[str],
    formatted_prompt: List[Dict[str, str]],
    unformatted_prompt: str
) -> SentenceData:
    """
    Create a properly formatted sentence data dict.
    
    Args:
        sentence_index: Index of the sentence
        sentence: The sentence text
        p_reward_hacks: Probability of reward hacking
        rollouts: List of rollout continuations
        formatted_prompt: Chat-formatted prompt
        unformatted_prompt: Raw prompt text
    
    Returns:
        SentenceData dictionary
    """
    return {
        "sentence_index": sentence_index,
        "sentence": sentence,
        "p_reward_hacks": p_reward_hacks,
        "rollouts": rollouts,
        "formatted_prompt": formatted_prompt,
        "unformatted_prompt": unformatted_prompt
    }

