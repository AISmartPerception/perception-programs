# InternVL3 BLINK Evaluation with Perception Programs

Evaluation framework for InternVL3 models on the BLINK benchmark, supporting multiple reasoning approaches including Perception Programs (PP), Chain-of-Thought (CoT), and visual tool-based reasoning.

## Repository Organization

### Main Evaluation Script
- **`icl_internvl3_blink.py`** - Main entry point for running evaluations on BLINK tasks

### Configuration Files (`eval_config/`)
Configuration files organized by evaluation mode:
- **`pp_config/`** - Perception Program configurations (flow paths, correspondence files, etc.)
- **`cot_config/`** - Chain-of-Thought configurations for specific tasks
- **`raw_tool_config/`** - Visual tool configurations (depth maps, flow visualizations, etc.)

Each config specifies task-specific paths and templates for data loading.

### Prompts (`prompts/`)
Modular prompt system with problem-specific customization:
- **`default/`** - Default system prompts and ICL examples
- **`{problem}/`** - Problem-specific prompts (e.g., `visual_correspondence/`, `jigsaw/`, etc.)
- **`loader.py`** - Automatic prompt loading with fallback to defaults

Each problem directory contains:
- `system_prompts.py` - System prompts for PP/CoT/Tool modes
- `icl_examples.py` - In-context learning examples

### Utilities
- **`pp_tools/`** - Perception program generation and BLINK-specific evaluation utilities
- **`ml_utils/`** - Model utilities including batch inference helpers

### Examples (`examples/`)
Documentation and example prompts showing reasoning strategies.

## Setup

Activate the conda environment:
```bash
# If the environment does not exist yet:
conda create -n <env-name> python=3.10 -y

# Activate the environment:
conda activate <env-name>
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic evaluation command:
```bash
python icl_internvl3_blink.py \
  --problem multi_view_reasoning \
  --mode pp \
  --model_size 4b \
  --batch_size 8
```

**Arguments:**
- `--problem` - Task to evaluate (e.g., `multi_view_reasoning`, `visual_correspondence`, `jigsaw`, `object_localization`)
- `--mode` - Reasoning approach: `pp` (Perception Programs), `cot` (Chain-of-Thought), or `tool` (Visual Tools)
- `--model_size` - Model variant: `1b`, `2b`, or `4b`
- `--batch_size` - Batch size for inference (default: 8)
- `--config` - Optional path to custom config file (auto-selected if not provided)

## Supported Tasks

- Multi-view Reasoning (camera motion prediction)
- Visual Correspondence (point matching)
- Relative Depth (3/4/5 point variants)
- Jigsaw (puzzle piece matching)
- Semantic Correspondence
- Object Localization

## Output

Results are saved to `./output/{problem}/` as JSONL files containing predictions, ground truth, and full model outputs.

