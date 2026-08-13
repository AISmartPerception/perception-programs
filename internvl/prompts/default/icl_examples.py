"""
Default ICL (In-Context Learning) examples for different modes.
These are used as fallbacks when problem-specific examples don't exist.
"""

ICL_EXAMPLE_TOOL = """# PROBLEM-SPECIFIC INSTRUCTIONS
You are provided with images and supplementary visual analysis.
Use the provided information as your primary source of truth.
When answering, include your reasoning and present your final answer as \\boxed{(X)}.
""".strip()

ICL_EXAMPLE_PP = """# PROBLEM-SPECIFIC INSTRUCTIONS
You are provided with a perception program containing structured visual analysis.
Use this information to answer the question.
Format your answer as \\boxed{(X)}.
""".strip()

