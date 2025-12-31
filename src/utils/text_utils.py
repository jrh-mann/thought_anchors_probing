"""
Text processing utilities for sentence splitting.
"""

import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex on '. ' pattern.
    
    Simple initial implementation - can be refined later based on actual usage.
    
    Args:
        text: Input text to split
    
    Returns:
        List of sentences (including the period and space if they were present)
    """
    if not text or not text.strip():
        return []
    
    # Split on '. ' pattern
    # This is a simple approach - we keep the period with the sentence
    sentences = re.split(r'(\. )', text)
    
    # Recombine sentences with their periods
    result = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i + 1] == '. ':
            # Sentence + period
            result.append(sentences[i] + sentences[i + 1])
            i += 2
        else:
            # Last piece or no period
            if sentences[i].strip():
                result.append(sentences[i])
            i += 1
    
    # If no sentences were found, return the whole text as one sentence
    if not result:
        return [text]
    
    return result


def get_sentence_prefix(sentences: List[str], sentence_index: int) -> str:
    """
    Get the prefix up to and including the sentence at the given index.
    
    Args:
        sentences: List of sentences
        sentence_index: Index of the sentence to include (0-based)
    
    Returns:
        Concatenated prefix text
    """
    if sentence_index < 0 or sentence_index >= len(sentences):
        return ""
    
    return "".join(sentences[:sentence_index + 1])