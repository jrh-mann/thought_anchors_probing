"""
Chat formatting utilities for prompt handling.
Converts between unformatted (raw text) and formatted (messages array) representations.
"""

from typing import List, Dict, Optional


def concatenate_prompt(task: str, rules: str, grader: str) -> str:
    """
    Concatenate task, rules, and grader into a single prompt string.
    
    Args:
        task: Task description
        rules: Rules for the task
        grader: Grader code
    
    Returns:
        Concatenated prompt string
    """
    parts = [task.strip(), rules.strip(), grader.strip()]
    return "\n\n".join(parts)


def format_chat_messages(unformatted_prompt: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Format an unformatted prompt into chat message format.
    
    Args:
        unformatted_prompt: Raw text prompt
        system_message: Optional system message to prepend
    
    Returns:
        List of message dicts with 'role' and 'content'
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": unformatted_prompt})
    return messages


def extract_assistant_response(response: Dict, extract_content_func) -> Optional[str]:
    """
    Extract assistant response from API response.
    
    Args:
        response: API response dict
        extract_content_func: Function to extract content (e.g., client.extract_content)
    
    Returns:
        Assistant response text or None
    """
    return extract_content_func(response)


def messages_to_prompt_string(messages: List[Dict[str, str]], tokenizer=None) -> str:
    """
    Convert chat messages to a prompt string for vLLM offline inference.
    
    vLLM's LLM.generate() can accept prompt strings. For chat models, if a tokenizer
    is provided, we use its chat template. Otherwise, we format as a simple text prompt.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Optional tokenizer with apply_chat_template method
    
    Returns:
        Formatted prompt string
    """
    # If tokenizer is available and has chat template, use it
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            # Convert messages format: vLLM expects messages in OpenAI format
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception:
            # Fall back to simple formatting if chat template fails
            pass
    
    # Simple text formatting fallback
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(prompt_parts)