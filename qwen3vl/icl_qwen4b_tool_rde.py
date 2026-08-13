import random, json, os, time, re
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch, jsonlines, cv2
import generate_perception_program as pp
import numpy as np
from PIL import Image

mode = "Thinking"  # or "Instruct"
random.seed(42)
torch.manual_seed(42)

model_name = f"Qwen/Qwen3-VL-8B-{mode}"
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

def build_conversation(sample, k, depth_img):
    system_ = """
        You are an oracle answering multiple-choice questions about images. You will be provided an image and its corresponding depth map, you should use both to answer. In the depth map, white points are in the front and black ones are in the back. In your answer, only output the letter enclosed by parenthesis---i.e. (A)---without any prose.
        Output Rules:
            - Start with a short thinking section enclosed by <think> and </think> tags.
            - Provide a single final choice as (X) where X is A, B, C, etc.
            - Include, before the final choice, one short justification (maximum 20 words).
            - Do not include extra thoughts, <think> tags, step-by-step reasoning, chain-of-thought reasoning, etc.
    """
    icl_3p_example = """
        You are given an image with multiple points circled on, and a depth map. Using the image and depth map decide which point is the closest to the camera?

        One demonstrative example of how to use the depth map to solve a certain problem is given below.

        Images: (scene with three labeled points A, B, C) and its depth map (white = closer).
        Depth observations:
        - Region at A is the brightest among A, B, C.
        - Region at C is darker than A.
        - Region at B is the darkest, far in the scene.
        Conclusion: A corresponds to the closest depth.
        Answer: (A)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """

    icl_4p_example = """
        You are given an image with multiple points circled on, and a depth map. Using the image and depth map decide which point is the closest to the camera?

        One demonstrative example of how to use the depth map to solve a certain problem is given below.

        Images: (scene with four labeled points A, B, C, D) and its depth map (white = closer).
        Depth observations:
        - A lies on the brightest region among A, B, C, D.
        - D is medium gray and behind A.
        - C is darker than A and D.
        - B is darkest on the distant wagon.

        Conclusion: A is closest to the camera.
        Answer: (A)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """

    icl_5p_example = """
        You are given an image with multiple points circled on, and a depth map. Using the image and depth map decide which point is the closest to the camera?

        One demonstrative example of how to use the depth map to solve a certain problem is given below.

        Images: (scene with five labeled points A, B, C, D, E) and its depth map (white = closer).
         Depth observations:
            - B sits on the brightest region among A, B, C, D, E.
            - A and D are medium gray (closer than C/E but not brightest).
            - C and E are darker, indicating farther distance.

        Conclusion: B is closest to the camera.
        Answer: (B)

        Important Note: The answer above belongs to the example and must not be copied.
        You will now receive a new question. For it compute your own choice and output the answer.
    """

    if k == 3:
        icl_example = icl_3p_example
    elif k == 4:
        icl_example = icl_4p_example
    else:
        icl_example = icl_5p_example

    system = '\n'.join([system_, icl_example])
    img = Image.open(sample["image"])
    prompt = sample["question"]

    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "image", "image": depth_img},
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

def get_batched_tool_outputs(idxs, dataset):
    batched_depth_maps = []
    for i in idxs:
        depth_im_path = re.sub(
            r"(/blink\d+pointscenter/)images/(.+?)\.png$",
            r"\1depth/\2_depth.png",
            dataset[i]["image"]
        ).replace("_depth", "")
        depth_img = cv2.cvtColor(cv2.imread(depth_im_path), cv2.COLOR_BGR2RGB)
        batched_depth_maps.append(depth_img)
    return batched_depth_maps

def run_qwen_pp_batched(output_root, data,
                        batch_size=BATCH_SIZE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        max_tries=MAX_TRIES,
                        retry_factor=RETRY_NEW_TOKENS_FACTOR,
                        max_tokens_cap=MAX_NEW_TOKENS_CAP):
    os.makedirs(output_root, exist_ok=True)
    for k in data.keys():
        print(f"Processing HardBLINK-{k}!")
        dataset = data[k] # either 3, 4, 5
        out_path = os.path.join(output_root, f"hardblink_{k}_qwen3_8B_{mode.lower()}_faster.jsonl")
        jsonl_path = f"/data00/hugo/hardblink/blink{k}pointscenter/point_positions.jsonl"
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
                depth_img = get_batched_tool_outputs(idxs, dataset)
                conversations = [build_conversation(samples[j], k, depth_img[j]) for j in range(len(samples))]
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

def fetch_hardblink_data():
    all_n_choices = [3,4,5]
    data = {}
    root = "/data00/hugo/hardblink/"
    for n_choices in all_n_choices:
        task = f"blink{n_choices}pointscenter"
        data[n_choices] = []
        with jsonlines.open(os.path.join(root, task, f"blink{n_choices}pointscenter.jsonl")) as reader:
            for obj in reader:
                obj["idx"] = f"val_hardblink{n_choices}_{obj['image'].split('.')[0]}"
                obj["image"] = os.path.join(root, task, "images", obj["image"])
                obj["answer"] = f"({obj['answer']})"
                data[n_choices].append(obj)
    return data

if __name__ == "__main__":
    output_root = f"/data00/depth/open-source-models/qwen8b-{mode.lower()}/tool/"
    dataset = fetch_hardblink_data()
    run_qwen_pp_batched(output_root, dataset)