"""
Prompts for hardblink5pointscenter (Relative Depth Estimation with 5 points)
"""

from .system_prompts import SYSTEM_PROMPT_PP, SYSTEM_PROMPT_TOOL, SYSTEM_PROMPT_COT
from .icl_examples import ICL_EXAMPLE_PP, ICL_EXAMPLE_TOOL

__all__ = [
    'SYSTEM_PROMPT_PP',
    'SYSTEM_PROMPT_TOOL', 
    'SYSTEM_PROMPT_COT',
    'ICL_EXAMPLE_PP',
    'ICL_EXAMPLE_TOOL',
]

