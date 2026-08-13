from datasets import load_dataset
import random, sys
sys.path.append('..')
import numpy as np
import generate_perception_program as pp
from tqdm import tqdm
import json, os, time, re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

mode = "Thinking"
dataset = load_dataset("BLINK-Benchmark/BLINK", "Object_Localization",
                       cache_dir="/data00/mllm_datasets/BLINK")["val"]
random.seed(42)
torch.manual_seed(42)
dataset_size = len(dataset)
root = "/data00/object_localization/"

model_name = f"Qwen/Qwen3-VL-4B-{mode}"
target_dir = "/data00/multiview_reasoning/open-source-models/"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 8192
MAX_NEW_TOKENS_CAP = 8192
MAX_TRIES = 1
RETRY_NEW_TOKENS_FACTOR = 1.0

processor = AutoProcessor.from_pretrained(model_name,
                                          cache_dir=target_dir)
processor.tokenizer.padding_side = "left"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    cache_dir=target_dir,
).eval()

def extract_choice(text: str):
    m = re.findall(r'(?<!\w)\(\s*([ABCDEabcde])\s*\)(?!\w)', text)
    return f"({m[-1].upper()})" if m else ""

def build_conversation(sample, tool_img):
    system_ = """
        You are an oracle answering multiple-choice questions about images. You will be provided an image and its corresponding object localization map, you should use both to answer. In the object localization map, a correct bounding box is drawn over the object. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
        Output Rules:
            - Start with a short thinking section enclosed by <think> and </think> tags.
            - Provide a single final choice as (X) where X is A, B, C, etc.
            - Include, before the final choice, one short justification (maximum 20 words).
            - Do not include extra thoughts, <think> tags, step-by-step reasoning, chain-of-thought reasoning, etc.
    """
    icl_example = """
        You are given an image and two candidate bounding boxes are drawn: Box A (green) and Box B (red)
        You are also given a object localization map that highlights the region corresponding to the region of interest specified in the query.
        Your goal is to decide which bounding box encloses the relevant object (from the query) better. 
        
        One demonstrative example of how to use tool output to solve a certain problem is given below.

        Observation from the tool:
        - The tool's tank-top region starts a bit further to the LEFT, covering the left shoulder/strap area.
        - The tool's region does NOT extend far to the RIGHT into the arm/racket area.

        Compare candidates:
        - Box B also starts left and covers that same left strap/shoulder area that the tool marks as tank top.
        - Box B stops roughly where the tool's tank-top region stops on the right, so it does not include too much of the arm.
        - Box A, in contrast, is shifted to the right: it misses part of the left tank-top area that the tool shows, and it extends farther right into areas the tool did not label as tank top.

        Therefore, the candidate that best matches the tool-labeled tank-top area is Box B.
        Answer: (B)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer. 
    """
    system = '\n'.join([system_, icl_example])
    img1 = sample["image_1"]
    tool_img_loaded = Image.open(tool_img).convert('RGB')
    prompt = sample["prompt"] + "\n" + sample["question"]

    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": tool_img_loaded},
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
        num_beams=1,
        max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
    outputs = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs

def get_batched_perception_programs(idxs, tool_path):
    batched_tools = []
    for i in idxs:
        img_id = dataset[i]['idx']
        tool_img_path = f"{tool_path}{img_id}.png"
        batched_tools.append(tool_img_path)
    return batched_tools

def run_qwen_with_pp_batched(output_root,
                             batch_size=BATCH_SIZE,
                             max_new_tokens=MAX_NEW_TOKENS,
                             max_tries=MAX_TRIES,
                             retry_factor=RETRY_NEW_TOKENS_FACTOR,
                             max_tokens_cap=MAX_NEW_TOKENS_CAP):
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"qwen3_4B_{mode.lower()}.jsonl")
    tool_path = "/data00/hugo/blink_fields/object_localization/detections/"
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
            tools_imgs_outputs = get_batched_perception_programs(idxs, tool_path)
            conversations = [build_conversation(samples[j], tools_imgs_outputs[j]) for j in range(len(samples))]

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
    output_root = f"/data00/object_localization/open-source-models/qwen4b-{mode.lower()}/tool/"
    run_qwen_with_pp_batched(output_root)