"""
Activation extraction and loading module.
"""

from .extract import store_activations
from .utils import reconstruct_sequence_messages, format_with_chat_template
from .load import (
    load_single_activation,
    load_metadata,
    load_prompt_activations,
    load_all_activations,
    stack_activations_by_layer,
    get_activation_matrix,
    get_reward_hacking_labels,
    quick_load,
    quick_load_legacy,
    clear_cache
)

__all__ = [
    'store_activations',
    'reconstruct_sequence_messages',
    'format_with_chat_template',
    'load_single_activation',
    'load_metadata',
    'load_prompt_activations',
    'load_all_activations',
    'stack_activations_by_layer',
    'get_activation_matrix',
    'get_reward_hacking_labels',
    'quick_load',
    'quick_load_legacy',
    'clear_cache'
]