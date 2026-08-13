"""
Prompt loader with fallback mechanism.

This module loads prompts specific to a problem, with automatic fallback to defaults
when problem-specific prompts are not available.
"""

import importlib
import os
from typing import Optional


class PromptLoader:
    """
    Loads prompts with problem-specific customization and default fallback.
    
    Usage:
        loader = PromptLoader(problem='visual_correspondence')
        system_prompt = loader.get_system_prompt('pp')
        icl_example = loader.get_icl_example('tool')
    """
    
    def __init__(self, problem: Optional[str] = None):
        """
        Initialize the prompt loader.
        
        Args:
            problem: Name of the specific problem (e.g., 'visual_correspondence').
                    If None or if problem-specific prompts don't exist, defaults are used.
        """
        self.problem = problem
        self._system_prompts_cache = {}
        self._icl_examples_cache = {}
        
    def _load_module(self, module_name: str, problem: Optional[str] = None):
        """
        Load a prompt module with fallback to default.
        
        Args:
            module_name: Name of the module ('system_prompts' or 'icl_examples')
            problem: Problem name for problem-specific prompts
            
        Returns:
            The loaded module, or None if not found
        """
        # Try problem-specific first
        if problem:
            try:
                return importlib.import_module(f'prompts.{problem}.{module_name}')
            except (ImportError, ModuleNotFoundError):
                pass
        
        # Fallback to default
        try:
            return importlib.import_module(f'prompts.default.{module_name}')
        except (ImportError, ModuleNotFoundError):
            return None
    
    def get_system_prompt(self, mode: str) -> str:
        """
        Get system prompt for a given mode.
        
        Args:
            mode: One of 'pp', 'tool', or 'cot'
            
        Returns:
            The system prompt string
            
        Raises:
            ValueError: If mode is invalid or prompt not found
        """
        mode = mode.lower()
        if mode not in ['pp', 'tool', 'cot']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of 'pp', 'tool', 'cot'")
        
        cache_key = f"{self.problem}:{mode}"
        if cache_key in self._system_prompts_cache:
            return self._system_prompts_cache[cache_key]
        
        # Load the module
        module = self._load_module('system_prompts', self.problem)
        if module is None:
            raise ValueError(f"Could not load system prompts for problem '{self.problem}'")
        
        # Get the appropriate prompt
        prompt_name = f"SYSTEM_PROMPT_{mode.upper()}"
        if not hasattr(module, prompt_name):
            raise ValueError(f"Prompt '{prompt_name}' not found in module")
        
        prompt = getattr(module, prompt_name)
        self._system_prompts_cache[cache_key] = prompt
        return prompt
    
    def get_icl_example(self, mode: str) -> str:
        """
        Get ICL example for a given mode.
        
        Args:
            mode: One of 'pp' or 'tool' (cot doesn't use ICL examples)
            
        Returns:
            The ICL example string
            
        Raises:
            ValueError: If mode is invalid or example not found
        """
        mode = mode.lower()
        if mode not in ['pp', 'tool']:
            raise ValueError(f"Invalid mode for ICL examples: {mode}. Must be one of 'pp', 'tool'")
        
        cache_key = f"{self.problem}:{mode}"
        if cache_key in self._icl_examples_cache:
            return self._icl_examples_cache[cache_key]
        
        # Load the module
        module = self._load_module('icl_examples', self.problem)
        if module is None:
            raise ValueError(f"Could not load ICL examples for problem '{self.problem}'")
        
        # Get the appropriate example
        example_name = f"ICL_EXAMPLE_{mode.upper()}"
        if not hasattr(module, example_name):
            raise ValueError(f"Example '{example_name}' not found in module")
        
        example = getattr(module, example_name)
        self._icl_examples_cache[cache_key] = example
        return example
    
    def get_all_prompts(self, mode: str) -> dict:
        """
        Get all prompts for a given mode.
        
        Args:
            mode: One of 'pp', 'tool', or 'cot'
            
        Returns:
            Dictionary with 'system_prompt' and optionally 'icl_example' keys
        """
        result = {
            'system_prompt': self.get_system_prompt(mode)
        }
        
        # ICL examples only for pp and tool modes
        if mode in ['pp', 'tool']:
            result['icl_example'] = self.get_icl_example(mode)
        
        return result

