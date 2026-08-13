# Qwen3VL Experiments

Standalone eval scripts running Qwen3-VL-4B (Thinking) on BLINK tasks, one
script per task per mode. Each script loads the model, loads its BLINK
subset, and runs inference over the val split.

## Naming

- `qwen4b_{task}.py` — Standard mode (images only, no tool info)
- `icl_qwen4b_tool_{task}.py` — Tool mode (raw tool output, e.g. a flow map
  image, passed alongside the images)
- `icl_qwen4b_pp_{task}.py` — PP mode (tool output converted to a
  `<perceptionprogram>` text block via `generate_perception_program.py`)

## Tasks

`mvr` (multi-view reasoning), `viscorr` (visual correspondence), `rde`
(relative depth), `jigsaw`, `semantic_correspondence`, `obj_loc` (object
localization).

## Usage

Each file is self-contained — edit the dataset/cache paths and model
`target_dir` at the top, then run directly:

```bash
python icl_qwen4b_pp_mvr.py
```
