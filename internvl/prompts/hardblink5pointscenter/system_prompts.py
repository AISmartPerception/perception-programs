"""
System prompts for hardblink5pointscenter (Relative Depth Estimation with 5 points).
Uses unified relative_depth prompts.
"""

from prompts.relative_depth.system_prompts import (
    SYSTEM_PROMPT_PP,
    SYSTEM_PROMPT_TOOL,
    SYSTEM_PROMPT_COT
)

__all__ = ['SYSTEM_PROMPT_PP', 'SYSTEM_PROMPT_TOOL', 'SYSTEM_PROMPT_COT']
