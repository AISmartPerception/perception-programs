"""
Prompts for Object Localization task.
"""

from .system_prompts import SYSTEM_PROMPT_COT, SYSTEM_PROMPT_PP, SYSTEM_PROMPT_TOOL
from .icl_examples import ICL_EXAMPLE_PP, ICL_EXAMPLE_TOOL

__all__ = [
    'SYSTEM_PROMPT_COT',
    'SYSTEM_PROMPT_PP',
    'SYSTEM_PROMPT_TOOL',
    'ICL_EXAMPLE_PP',
    'ICL_EXAMPLE_TOOL',
]

