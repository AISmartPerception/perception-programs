from datasets import load_dataset
import random, json, os, time, re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

mode = "Thinking"  # or "Instruct"
dataset = load_dataset("BLINK-Benchmark/BLINK", "Jigsaw",
                       cache_dir="/data00/mllm_datasets/BLINK")["val"]
random.seed(42)
torch.manual_seed(42)

model_name = f"Qwen/Qwen3-VL-4B-{mode}"
target_dir = "/data00/multiview_reasoning/open-source-models/"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 8192
MAX_NEW_TOKENS_CAP = 32768
MAX_TRIES = 3
RETRY_NEW_TOKENS_FACTOR = 2.0

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

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def build_conversation(sample, stitched_one, stitched_two):
    system_ = """
        You are an oracle answering multiple-choice questions about images. You are also given two stitched outputs where each missing part is filled in. Use all the images as much as possible to answer. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
    """
    icl_example = """
        You are given an image with the lower-right corner missing, two candidate pieces, and two stitched previews showing how each candidate would look if placed in the missing spot. Decide which candidate fits best with the original image.
        (A) the second image  (B) the third image

        One example of how to use the stitched previews to solve the problem is given below.

        Images: (main image with missing corner, candidates A & B, and two stitched previews)
        Tool output example: the second stitched image (with candidate B) aligns perfectly with the skyline and perspective.
        Answer: (B)
        
        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """
    system = '\n'.join([system_, icl_example])

    img1 = sample["image_1"]
    img2 = sample["image_2"]
    img3 = sample["image_3"]
    prompt = sample['prompt'] + "\n" + sample["question"]

    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "image", "image": img3},
                {"type": "image", "image": stitched_one},
                {"type": "image", "image": stitched_two},
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
    batched_stitched_ones = []
    batched_stitched_twos = []
    for i in idxs:
        img_id = dataset[i]['idx']
        stitched_one = Image.open(tool_outputs + img_id + "_img1.png")
        stitched_two = Image.open(tool_outputs + img_id + "_img2.png")
        batched_stitched_ones.append(stitched_one)
        batched_stitched_twos.append(stitched_two)
    return batched_stitched_ones, batched_stitched_twos

def run_qwen_tool_batched(output_root,
                          batch_size=BATCH_SIZE,
                          max_new_tokens=MAX_NEW_TOKENS,
                          max_tries=MAX_TRIES,
                          retry_factor=RETRY_NEW_TOKENS_FACTOR,
                          max_tokens_cap=MAX_NEW_TOKENS_CAP):
    os.makedirs(output_root, exist_ok=True)
    out_path = os.path.join(output_root, f"qwen3_4B_{mode.lower()}.jsonl")
    tool_outputs = "/data00/jigsaw/expert_outputs_for_pp/tool_outputs/"
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
            stitched_ones, stitched_twos = get_batched_tool_outputs(idxs, tool_outputs)
            conversations = [build_conversation(samples[j],
                                                stitched_ones[j],
                                                stitched_twos[j]) for j in range(len(samples))]
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
    output_root = f"/data00/jigsaw/open-source-models/qwen4b-{mode.lower()}/tool/"
    run_qwen_tool_batched(output_root)