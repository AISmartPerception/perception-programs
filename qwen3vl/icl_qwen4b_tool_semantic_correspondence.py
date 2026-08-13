from datasets import load_dataset
import random, sys
sys.path.append('..')
from tqdm import tqdm
import json, os, time, re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

mode = "Thinking"
dataset = load_dataset("BLINK-Benchmark/BLINK", "Semantic_Correspondence",
                       cache_dir="/data00/mllm_datasets/BLINK")["val"]
random.seed(42)
torch.manual_seed(42)
dataset_size = len(dataset)
root = "/data00/multiview_reasoning/"

model_name = f"Qwen/Qwen3-VL-4B-{mode}"
target_dir = "/data00/multiview_reasoning/open-source-models/"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 8192
MAX_NEW_TOKENS_CAP = 8192
MAX_TRIES = 1
RETRY_NEW_TOKENS_FACTOR = 1.0

processor = AutoProcessor.from_pretrained(model_name, cache_dir=target_dir)
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

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def build_conversation(sample, tool_output):
    system_ = """
        You are an oracle answering multiple-choice questions about images. You will be provided two images (source and target) and corresponding similarity density map overlayed on target image. In density map, brighter colors indicate higher similarity. Use the similarity map and images to answer. In the visual correspondence map (left image is image 1, right image is image 2), each line connects two similar points in the two images. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
        Output Rules:
            - Start with a short thinking section enclosed by <think> and </think> tags.
            - Provide a single final choice as (X) where X is A, B, C, etc.
            - Include, before the final choice, one short justification (maximum 20 words).
            - Do not include extra thoughts, <think> tags, step-by-step reasoning, chain-of-thought reasoning, etc.
    """
    icl_example = """
        You are given two images from the same object category and corresponding similarity density map. A reference point (REF) is marked on the first image. 
        On the second image, four candidate points A-D are marked. 
        Choose which point corresponds to the reference point.
        (A) Point A  (B) Point B  (C) Point C  (D) Point D

        One example of how to use the auxiliary visualization to solve the problem is given below.

        Images: (reference image with REF, candidate image with A-D, and a score-based density overlay on the candidate image)
        Tool output example: the density map peaks along the bird's mid-leg near point C, ruling out the beak (A) and the foot (B); point D is higher at the hip.
        Answer: (C)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """
    system = '\n'.join([system_, icl_example])

    img1 = sample["image_1"]
    img2 = sample["image_2"]

    prompt = sample['prompt'] + "\n" + sample["question"]
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "image", "image": tool_output},
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

def get_batched_tool_outputs(idxs, tool_outputs):
    batched_tool_output = []
    for i in idxs:
        img_id = dataset[i]['idx']
        tool_output_img = f"{tool_outputs}{img_id}_tool.png"
        tool_output_img_read = Image.open(tool_output_img).convert('RGB')
        batched_tool_output.append(tool_output_img_read)
    return batched_tool_output

def run_qwen_tool_batched(output_root,
                          batch_size=BATCH_SIZE,
                          max_new_tokens=MAX_NEW_TOKENS,
                          max_tries=MAX_TRIES,
                          retry_factor=RETRY_NEW_TOKENS_FACTOR,
                          max_tokens_cap=MAX_NEW_TOKENS_CAP):
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"qwen3_4B_{mode.lower()}.jsonl")
    tool_outputs = "/data00/semantic_correspondence/tool_out_imgs/"
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
            batched_tool_outs = get_batched_tool_outputs(idxs, tool_outputs)
            conversations = [build_conversation(samples[j], batched_tool_outs[j]) for j in range(len(samples))]
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
    print(f"Final Accuracy: {acc} ({correct}/{len(final_list)})")
    print(f"Unanswered after retries: {len([r for r in final_list if r['prediction']==''])}")
    print(f"Total Time Taken: {total_time_end-total_time_st}s")
    
    with open(out_path, "w") as f:
        for r in final_list:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(final_list)} results → {out_path}")

if __name__ == "__main__":
    output_root = f"/data00/semantic_correspondence/open-source-models/qwen4b-{mode.lower()}/tool/"
    run_qwen_tool_batched(output_root)