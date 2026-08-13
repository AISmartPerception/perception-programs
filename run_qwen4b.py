"""
    Code to run Qwen-4B on Object Localization Task from BLINK.
    This code runs Perception Program with Qwen3VL-4B.
"""

from datasets import load_dataset
import random
import numpy as np
import generate_perception_program as pp
from tqdm import tqdm
import json, os, time, re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

mode = "Thinking" # "Instruct"
dataset = load_dataset("BLINK-Benchmark/BLINK", "Object_Localization",
                       cache_dir="/mllm_datasets/BLINK")["val"]
random.seed(42)
torch.manual_seed(42)
dataset_size = len(dataset)

model_name = f"Qwen/Qwen3-VL-4B-{mode}"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 8192
MAX_NEW_TOKENS_CAP = 8192
MAX_TRIES = 1
RETRY_NEW_TOKENS_FACTOR = 1.0

processor = AutoProcessor.from_pretrained(model_name)
processor.tokenizer.padding_side = "left"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
).eval()

def extract_choice(text: str):
    m = re.findall(r'(?<!\w)\(\s*([ABCDEabcde])\s*\)(?!\w)', text)
    return f"({m[-1].upper()})" if m else ""

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def format_annot_coords(coords):
    formatted_coords = []  
    for coord in coords:
        label = coord['label']  
        if '_' in label:  
            bbox_id, position = label.split('_', 1)  
            position_formatted = position.replace('_', '-')  
            new_label = f"Bounding box {bbox_id}, {position_formatted} corner"
        else: 
            new_label = label  
        
        formatted_coords.append({  
            'x': int(round(coord['x'])),
            'y': int(round(coord['y'])),
            'label': new_label  
        })
    return formatted_coords

def coords_to_perceptionprogram(coords):
    coords = sorted(coords, key=lambda x: x["label"])
    lines = []
    lines.append("<perceptionprogram>")
    lines.append("modality: point-detection")
    lines.append("granularity: pixels")
    lines.append('note: "Each item lists one of the points relative to this task in the following format:')
    lines.append("- c: coordinates of the point")
    lines.append("- b: label of the point")
    lines.append('Coordinates are normalized between (0, 0) in the upper left and (1000, 1000) in the bottom right."')
    lines.append("items:")
    for pt in coords:
        lines.append(f"  - c: ({pt['x']}, {pt['y']})")
        lines.append(f"    b: \"{pt['label']}\"")
    lines.append("</perceptionprogram>")
    return "\n".join(lines)

def fetch_obj_loc_pp(det_path):
    obj_loc_detections = {}
    with open(det_path, 'r') as f:
        for line in f:  
            det = json.loads(line)
            obj_loc_detections[det['idx']] = det
    return obj_loc_detections

def build_conversation(sample, det_pp):
    system_ = """
        You are an oracle answering multiple-choice questions about images. You will receive extra visual information or instructions in textual format to assist you in answering. They will be enclosed in <perceptionprogram></perceptionprogram> tags. Use this information as much as possible to get more precise answers. 
        Output Rules:
            - Start with a short thinking section enclosed by <think> and </think> tags.
            - Provide a single final choice as (X) where X is A, B, C, etc.
            - Include, before the final choice, one short justification (maximum 20 words).
            - Do not include extra thoughts, <think> tags, step-by-step reasoning, chain-of-thought reasoning, etc.
    """
    icl_example = """
        You are given an image with different bounding boxes covering the region of interest.
        <perceptionprogram> blocks are provided listing the corner points of each candidate bounding box and detected objects relevant to the query.
        Your goal is to decide which bounding box encloses the relevant object (from the query) better. 

        One demonstrative example of how to use perception program to solve a certain problem is given below.

        Images: (Single image with tank-top; two candidate boxes: A (green), B (red))
        Perception: The perception programs report:
        - Points -> A: (x0,y0)=(346,369), (x1,y1)=(631,833)
                    B: (290,346)-(570,859).
        - Object detection for 'tank top' => [285,343,578,860].
        Box B closely matches the detected bounds (left approx. 290 vs 285, right approx. 570 vs 578), while Box A is shifted right—its left edge (346) cuts into the left strap/side and its right edge (631) includes extra arm/background. Vertically both are comparable, but B is tighter and more faithful to the outermost garment pixels.
        Answer: (B)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.    
    """
    system = '\n'.join([system_, icl_example])

    img1 = sample["image_1"]
    question_ = sample["prompt"] + "\n" + sample["question"]
    prompt = '\n'.join([question_, det_pp])

    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "text",  "text": prompt.strip()},
            ],
        },
    ]

def make_batches(idxs, n):
    for i in range(0, len(idxs), n):
        yield idxs[i:i+n]

@torch.inference_mode()
def generate_batch(conversations, max_new_tokens):
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs

def get_batched_perception_programs(idxs, annotations, obj_loc_detections):
    batched_pps = []
    for i in idxs:
        img_id = dataset[i]['idx']
        formatted_coords_i = annotations[i]
        assert formatted_coords_i['idx'] == img_id
        formatted_coords = format_annot_coords(formatted_coords_i['coords'])
        coords_pp = coords_to_perceptionprogram(formatted_coords)

        det_data = obj_loc_detections[img_id]
        detections = det_data.get('detections', [])
        det_pp, _ = pp.emit_perception_program(
            modality="object-detection",  
            field=detections,
            seg=None, # irrelevant for object-det
            class_names=None, # irrelevant for object-det  
            grid=(10, 10),  # irrelevant for object-det
            add_relations=False, # irrelevant for object-det
            tau=0.08, # irrelevant for object-det
            relation_cap=500 # irrelevant for object-det
        )
        semantic_pp_combined = '\n'.join([coords_pp, det_pp]) 
        batched_pps.append(semantic_pp_combined)
    return batched_pps

def run_qwen_with_pp_batched(output_root,
                             batch_size=BATCH_SIZE,
                             max_new_tokens=MAX_NEW_TOKENS,
                             max_tries=MAX_TRIES,
                             retry_factor=RETRY_NEW_TOKENS_FACTOR,
                             max_tokens_cap=MAX_NEW_TOKENS_CAP):
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"qwen3_4B_{mode.lower()}.jsonl")

    annotations = load_jsonl("Object_Localization_val_annotations.jsonl") # point annotation files
    obj_loc_detections = fetch_obj_loc_pp("detections.jsonl") # detection results from LLMDet
    
    N = len(dataset)
    all_indices = list(range(N))
    results = {i: None for i in all_indices}
    
    attempt = 1
    pending = all_indices[:]
    wrong_counter = 0
    total_time_st = time.time()

    while pending and attempt <= max_tries:
        budget = int(min(max_new_tokens * (retry_factor ** (attempt - 1)), max_tokens_cap))
        pbar = tqdm(list(make_batches(pending, batch_size)),
                    desc=f"At: {attempt}/{max_tries} | Toks={budget}",
                    leave=False, dynamic_ncols=True)
        next_pending = []
        for idxs in pbar:
            samples = [dataset[i] for i in idxs]
            pps = get_batched_perception_programs(idxs, annotations, obj_loc_detections)
            conversations = [build_conversation(samples[j], pps[j]) for j in range(len(samples))]

            t0 = time.time()
            outputs = generate_batch(conversations, max_new_tokens=budget)
            t1 = time.time()

            for ds_idx, s, o in zip(idxs, samples, outputs):
                pred = extract_choice(o)
                rec = {
                    "image": s["idx"],
                    "question": s["question"],
                    "ground_truth": s["answer"],
                    "prediction": pred,
                    "output": o,
                    "correct": (pred == s["answer"]) if pred else False,
                    "attempt": attempt,
                    "max_new_tokens": budget,
                }
                results[ds_idx] = rec

                if pred == "":
                    next_pending.append(ds_idx)
                else:
                    wrong_counter += (not rec["correct"])

            pbar.set_postfix({"Pending": len(next_pending),
                              "WrongSoFar": wrong_counter,
                              "BatchTime": f"{t1 - t0:.2f}s"},
                             refresh=True)

        pending = next_pending
        attempt += 1

    total_time_end = time.time()

    final_list = [results[i] for i in sorted(results.keys()) if results[i] is not None]
    correct = sum(r["correct"] for r in final_list)
    acc = correct / len(final_list) if final_list else 0.0
    print(f"Final Accuracy: {acc:.4f} ({correct}/{len(final_list)})")
    print(f"Unanswered after retries: {len([r for r in final_list if r['prediction']==''])}")
    print(f"Total Time Taken: {total_time_end-total_time_st}s")
    
    with open(out_path, "w") as f:
        for r in final_list:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(final_list)} results → {out_path}")

if __name__ == "__main__":
    output_root = f"/object_localization/qwen4b-{mode.lower()}/pp/"
    run_qwen_with_pp_batched(output_root)